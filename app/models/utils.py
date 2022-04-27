from utils import class_utils


def import_model_class(data_cls_name):
    assert isinstance(data_cls_name, str)
    data_class = class_utils.import_class(f"models.{data_cls_name}")
    return data_class
