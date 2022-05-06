"""Flask web server serving text_recognizer predictions."""
import os
import logging
import json

from flask import Flask, request, jsonify
from flask_classful import FlaskView, route
from PIL import ImageStat

if __name__ == "__main__":
    from pathlib import Path
    import sys

    dir_app = Path(__file__).resolve().parents[2]
    sys.path.append(str(dir_app))
    print(dir_app)

from data_io import img_io


app = Flask(__name__)

class Flask_Img_Pred_Api_Server(FlaskView):

    def run(self, host, port, debug):
        assert isinstance(app, Flask)
        app.run(host=host, port=port, debug=debug)

    @route("/", methods=["GET"])
    def index(self):
        """Provide simple health check route."""
        return "Hello, world!"

    @route("/v1/predict", methods=["GET", "POST"])
    def predict(self):
        pass

    @classmethod
    def _load_image(cls):
        if request.method == "POST":
            data = request.get_json()
            data_d = json.loads(data)
            if data is None:
                return "no json received"
            return img_io.decode_b64_2_img_urlsafe(data_d["image"], grayscale=True)
        if request.method == "GET":
            image_url = request.args.get("image_url")
            if image_url is None:
                return "no image_url defined in query string"
            logging.info("url {}".format(image_url))
            return img_io.read_image_pil_file(image_url, grayscale=True)
        raise ValueError("Unsupported HTTP method")


def main():
    server = Flask_Img_Pred_Api_Server()
    server.run(host="0.0.0.0", port=8000, debug=False)  # nosec


if __name__ == "__main__":
    main()
