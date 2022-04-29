"""Flask web server serving text_recognizer predictions."""
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
from data_io import pathutils
from services.predicts.para_text_recognizer_pred import (
    Para_Text_Recognizer_Pred_Service,
)
from data_io.servers.flask_img_pred_api_server import ImgPredApiServer


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Do not use GPU
    project_name = "para_text_recognizer"
    dir_workspace, logger_py = workspace_io.create_workspace_predict(project_name)
    args_config = config_io.load_config_args_predict(project_name)

    logger_io.write_run_start(logger_py, project_name, dir_workspace, args_config)
    try:
        service_pred = Para_Text_Recognizer_Pred_Service(
            project_name, logger_py=logger_py, args_config=args_config
        )
        server = ImgPredApiServer()
        server.service_pred = service_pred
        server.run(host="0.0.0.0", port=8000, debug=False)  # nosec
    except Exception as e:
        logger_io.write_err_log(logger_py)


if __name__ == "__main__":
    main()
