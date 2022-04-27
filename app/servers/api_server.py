"""Flask web server serving text_recognizer predictions."""
import os
import logging

from flask import Flask, request, jsonify
from flask_classful import FlaskView, route
from PIL import ImageStat

if __name__ == "__main__":
    from pathlib import Path
    import sys

    dir_app = Path(__file__).resolve().parents[1]
    sys.path.append(str(dir_app))
    print(dir_app)


from data_io import img_io
from text_recognizer.paragraph_text_recognizer import ParagraphTextRecognizer


class ApiServer(FlaskView):

    app = Flask(__name__)

    def __init__(self):
        super().__init__()
        self._model = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model_n):
        assert hasattr(model_n, "predict")
        assert callable(getattr(model_n, "predict"))
        self._model = model_n

    def run(self):
        assert isinstance(self.app, Flask)
        self.app.run()

    @route("/")
    def index(self):
        """Provide simple health check route."""
        return "Hello, world!"

    @route("/v1/predict", methods=["GET", "POST"])
    def predict(self):
        """Provide main prediction API route. Responds to both GET and POST requests."""
        image = self._load_image()
        pred = self._model.predict(image)
        image_stat = ImageStat.Stat(image)
        logging.info("METRIC image_mean_intensity {}".format(image_stat.mean[0]))
        logging.info("METRIC image_area {}".format(image.size[0] * image.size[1]))
        logging.info("METRIC pred_length {}".format(len(pred)))
        logging.info("pred {}".format(pred))
        return jsonify({"pred": str(pred)})

    @staticmethod
    def _load_image(cls):
        if request.method == "POST":
            data = request.get_json()
            if data is None:
                return "no json received"
            return img_io.decode_b64_2_img(data["image"], grayscale=True)
        if request.method == "GET":
            image_url = request.args.get("image_url")
            if image_url is None:
                return "no image_url defined in query string"
            logging.info("url {}".format(image_url))
            return img_io.read_image_pil_file(image_url, grayscale=True)
        raise ValueError("Unsupported HTTP method")


ApiServer.register(ApiServer.app, route_base="/")


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Do not use GPU
    logging.basicConfig(level=logging.INFO)

    model = ParagraphTextRecognizer()
    server = ApiServer()
    server.model = model
    server.run(host="0.0.0.0", port=8000, debug=False)  # nosec


if __name__ == "__main__":
    main()
