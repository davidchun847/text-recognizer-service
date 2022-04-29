import argparse

from utils import class_utils


def import_model_class(model_name):
    assert isinstance(model_name, str)
    model_cls = class_utils.import_class(f"models.{model_name}")
    return model_cls


def import_model(model_name, config_data, arg):
    assert isinstance(model_name, str)
    assert isinstance(config_data, dict)
    assert isinstance(arg, argparse.Namespace)
    model_cls = import_model_class(model_name=model_name)
    model = model_cls(data_config=config_data, args=arg)
    return model