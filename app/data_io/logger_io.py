from pathlib import Path
from data_io import pathutils
import datetime
import logging
import traceback


def create_logger(log_name, dir, log_level=logging.DEBUG):
    assert isinstance(log_level, int)
    pathutils.check_is_dir(dir)
    name_file = pathutils.create_now_name("log") + ".log"
    path_log = Path(dir, name_file)

    formatter = logging.Formatter(
        "%(asctime)s:%(levelname)s:%(filename)s:%(lineno)d:%(message)s"
    )
    file_handler = logging.FileHandler(filename=path_log)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(log_name)
    logger.addHandler(file_handler)
    logger.setLevel(log_level)
    return logger


def print_log_info(logger, text):
    assert isinstance(logger, logging.Logger)
    assert isinstance(text, str)
    logger.info(text)
    print(text)


def write_run_start(logger, project_name, workspace=None, args=None):
    assert isinstance(logger, logging.Logger)
    assert isinstance(project_name, str)
    logger.info(f"project_name: {project_name}")
    curr_time = datetime.datetime.now()
    curr_time_str = curr_time.strftime("%Y,%m,%d:%H,%M,%S")
    logger.info(f"time run: {curr_time_str}")
    if workspace:
        logger.info(f"workspace: {workspace}")
    if args:
        logger.info(f"args:")
        logger.info(args)


def write_err_log(logger):
    assert isinstance(logger, logging.Logger)
    logger.error(traceback.format_exc())
    print(traceback.format_exc())
