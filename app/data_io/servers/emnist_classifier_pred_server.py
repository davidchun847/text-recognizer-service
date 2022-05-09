import os
import logging

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
from data_io import workspace_io
from data_io import config_io
from data_io import logger_io
from services.predicts.text_recognizer_pred_service import (
    Text_Recognizer_Pred_Service,
)
from data_io.servers.flask_img_pred_api_server import Flask_Img_Pred_Api_Server
from services.predictors.emnist_classifier_predictor_builder import (
    EMNIST_Classifier_Predictor_Builder,
)


try:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Do not use GPU
    project_name = "emnist_classifier"
    dir_workspace, logger_py = workspace_io.create_workspace_predict(project_name)
    args_config = config_io.load_config_args_predict(project_name)
    logger_io.write_run_start(logger_py, project_name, dir_workspace, args_config)
    service_pred = Text_Recognizer_Pred_Service(
        project_name,
        logger_py=logger_py,
        predictor_builder_cls=EMNIST_Classifier_Predictor_Builder,
        args_config=args_config,
    )
except Exception as e:
    logger_io.write_err_log(logger_py)

app = Flask(__name__)

class EMNIST_Classifier_Pred_Server(Flask_Img_Pred_Api_Server):


    def run(self, host, port, debug):
        assert isinstance(app, Flask)
        app.run(host=host, port=port, debug=debug)


    @route("/", methods=["GET"])
    def index(self):
        """Provide simple health check route."""
        return "Hello, world!, emnist_classifier_pred_server"


    @route("/v1/predict", methods=["GET", "POST"])
    def predict(self):
        super().predict()
        """Provide main prediction API route. Responds to both GET and POST requests."""
        image = self._load_image()
        assert callable(getattr(service_pred, "predict"))
        pred = service_pred.predict(image)
        image_stat = ImageStat.Stat(image)
        logging.info("METRIC image_mean_intensity {}".format(image_stat.mean[0]))
        logging.info("METRIC image_area {}".format(image.size[0] * image.size[1]))
        logging.info("METRIC pred_length {}".format(len(pred)))
        logging.info("pred {}".format(pred))
        return jsonify({"pred": str(pred)})


EMNIST_Classifier_Pred_Server.register(app, route_base='/')




def main():
    try:
        server = EMNIST_Classifier_Pred_Server()
        server.run(host="127.0.0.1", port=8000, debug=False)  # nosec
    except Exception as e:
        logger_io.write_err_log(logger_py)


if __name__ == "__main__":
    main()
