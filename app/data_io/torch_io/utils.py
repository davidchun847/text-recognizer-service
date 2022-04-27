import argparse

from utils import class_utils


def import_data_class(data_cls_name):
    assert isinstance(data_cls_name, str)
    data_class = class_utils.import_class(f"data_io.torch_io.{data_cls_name}")
    return data_class


def load_and_print_info(data_module_class) -> None:
    """Load EMNISTLines and print info."""
    parser = argparse.ArgumentParser()
    data_module_class.add_to_argparse(parser)
    args = parser.parse_args()
    dataset = data_module_class(args)
    dataset.prepare_data()
    dataset.setup()
    print(dataset)
