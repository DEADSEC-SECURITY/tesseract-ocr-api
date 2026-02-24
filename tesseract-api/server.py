from flask import Flask, request, jsonify
import subprocess
import tempfile
import os

app = Flask(__name__)

@app.route("/ocr", methods=["POST"])
def ocr():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    lang = request.form.get("lang", "eng")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["tesseract", tmp_path, "stdout", "-l", lang],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            return jsonify({"error": result.stderr}), 500
        return jsonify({"text": result.stdout.strip()})
    finally:
        os.unlink(tmp_path)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8884)
