import os
import base64
import io
import hashlib
from flask import Flask, request, jsonify, render_template, send_from_directory
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
ROOMS_DIR = os.path.join(os.path.dirname(__file__), "static", "saved_rooms")
os.makedirs(ROOMS_DIR, exist_ok=True)


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
    """Save room image to disk, deduplicated by content hash. Returns filename."""
    digest = hashlib.sha1(data).hexdigest()[:16]
    filename = f"{digest}.jpg"
    path = os.path.join(ROOMS_DIR, filename)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(data)
    return filename


@app.errorhandler(Exception)
def handle_exception(exc):
    return jsonify({"error": str(exc)}), 500


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/rooms", methods=["GET"])
def list_rooms():
    files = sorted(
        [f for f in os.listdir(ROOMS_DIR) if f.endswith(".jpg")],
        key=lambda f: os.path.getmtime(os.path.join(ROOMS_DIR, f)),
        reverse=True,  # newest first
    )
    return jsonify({"rooms": files})


@app.route("/rooms/<filename>")
def serve_room(filename):
    return send_from_directory(ROOMS_DIR, filename)


@app.route("/rooms/<filename>", methods=["DELETE"])
def delete_room(filename):
    path = os.path.join(ROOMS_DIR, filename)
    if os.path.exists(path):
        os.remove(path)
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

    # Save the room scene for future reuse
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
