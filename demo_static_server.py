from pathlib import Path

from flask import Flask, send_from_directory


ROOT = Path(__file__).resolve().parent
RUNTIME_ASSETS = ROOT / "runtime_assets"

app = Flask(__name__)


@app.get("/")
def index():
    return send_from_directory(RUNTIME_ASSETS, "cloud_realtime_demo.html")


@app.get("/models/<path:filename>")
def models(filename: str):
    return send_from_directory(RUNTIME_ASSETS, filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

