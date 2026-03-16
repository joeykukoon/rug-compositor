import os
import base64
import io
import hashlib
from flask import Flask, request, jsonify, render_template, send_from_directory, redirect
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

load_dotenv()

app = Flask(__name__)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

COMPOSITE_PROMPT = (
    "You are a photorealistic image compositor. "
    "I will give you two images: "
    "Image 1 is an overhead product shot of a rug. "
    "Image 2 is a room scene that already contains a rug on the floor. "
    "Your task: replace the existing rug in the room scene with the rug from Image 1. "
    "Match the perspective, lighting, shadows, and scale of the original rug precisely so the result looks photorealistic. "
    "Return only the composited room scene image with no text, annotations, or borders."
)

MAX_DIMENSION = 2048

# ---------------------------------------------------------------------------
# Storage — uses Cloudflare R2 when env vars are present, local disk otherwise
# ---------------------------------------------------------------------------

_R2_VARS = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME", "R2_PUBLIC_URL")
USE_R2 = all(os.environ.get(v) for v in _R2_VARS)

if USE_R2:
    import boto3
    from botocore.config import Config as BotoConfig

    _r2 = boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=BotoConfig(signature_version="s3v4"),
        region_name="auto",
    )
    _bucket = os.environ["R2_BUCKET_NAME"]
    _public_url = os.environ["R2_PUBLIC_URL"].rstrip("/")
else:
    ROOMS_DIR = os.path.join(os.path.dirname(__file__), "static", "saved_rooms")
    os.makedirs(ROOMS_DIR, exist_ok=True)


def storage_save(filename: str, data: bytes):
    if USE_R2:
        _r2.put_object(Bucket=_bucket, Key=filename, Body=data, ContentType="image/jpeg")
    else:
        path = os.path.join(ROOMS_DIR, filename)
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(data)


def storage_list() -> list[str]:
    """Return filenames, newest first."""
    if USE_R2:
        resp = _r2.list_objects_v2(Bucket=_bucket)
        objects = resp.get("Contents", [])
        objects.sort(key=lambda o: o["LastModified"], reverse=True)
        return [o["Key"] for o in objects]
    else:
        files = [f for f in os.listdir(ROOMS_DIR) if f.endswith(".jpg")]
        files.sort(key=lambda f: os.path.getmtime(os.path.join(ROOMS_DIR, f)), reverse=True)
        return files


def storage_delete(filename: str):
    if USE_R2:
        _r2.delete_object(Bucket=_bucket, Key=filename)
    else:
        path = os.path.join(ROOMS_DIR, filename)
        if os.path.exists(path):
            os.remove(path)


def storage_url(filename: str) -> str:
    if USE_R2:
        return f"{_public_url}/{filename}"
    else:
        return f"/rooms/{filename}"


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _read_upload(file_storage) -> tuple[bytes, str]:
    data = file_storage.read()
    mime = file_storage.mimetype or "image/jpeg"
    return data, mime


def _resize_for_api(data: bytes) -> tuple[bytes, str]:
    img = Image.open(io.BytesIO(data))
    img = img.convert("RGB")
    if max(img.size) > MAX_DIMENSION:
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue(), "image/jpeg"


def _bytes_to_base64_data_url(data: bytes, mime: str) -> str:
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _save_room(data: bytes) -> str:
    digest = hashlib.sha1(data).hexdigest()[:16]
    filename = f"{digest}.jpg"
    storage_save(filename, data)
    return filename


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.errorhandler(Exception)
def handle_exception(exc):
    return jsonify({"error": str(exc)}), 500


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/rooms", methods=["GET"])
def list_rooms():
    files = storage_list()
    rooms = [{"filename": f, "url": storage_url(f)} for f in files]
    return jsonify({"rooms": rooms})


@app.route("/rooms/<filename>")
def serve_room(filename):
    if USE_R2:
        return redirect(storage_url(filename))
    return send_from_directory(ROOMS_DIR, filename)


@app.route("/rooms/<filename>", methods=["DELETE"])
def delete_room(filename):
    storage_delete(filename)
    return jsonify({"ok": True})


@app.route("/process", methods=["POST"])
def process():
    if "rug" not in request.files or "room" not in request.files:
        return jsonify({"error": "Both 'rug' and 'room' images are required."}), 400

    rug_data, _ = _read_upload(request.files["rug"])
    room_data, _ = _read_upload(request.files["room"])
    prompt = request.form.get("prompt", "").strip() or COMPOSITE_PROMPT

    try:
        Image.open(io.BytesIO(rug_data)).verify()
        Image.open(io.BytesIO(room_data)).verify()
    except Exception:
        return jsonify({"error": "One or both uploaded files are not valid images."}), 400

    rug_data, rug_mime = _resize_for_api(rug_data)
    room_data, room_mime = _resize_for_api(room_data)

    saved_room_filename = _save_room(room_data)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=rug_data, mime_type=rug_mime),
                types.Part.from_bytes(data=room_data, mime_type=room_mime),
            ],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"]
            ),
        )
    except Exception as exc:
        return jsonify({"error": f"Gemini API error: {exc}"}), 502

    result_image_data_url = None
    result_text = None

    for part in response.candidates[0].content.parts:
        if part.inline_data is not None and not result_image_data_url:
            result_image_data_url = _bytes_to_base64_data_url(
                part.inline_data.data, part.inline_data.mime_type
            )
        elif part.text:
            result_text = part.text

    if result_image_data_url:
        return jsonify({
            "image": result_image_data_url,
            "text": result_text,
            "saved_room": saved_room_filename,
        })

    return jsonify(
        {"error": "Gemini did not return an image.", "details": result_text}
    ), 502


if __name__ == "__main__":
    app.run(debug=True)
