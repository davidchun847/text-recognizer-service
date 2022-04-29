import argparse

from utils import class_utils


def import_data_class(data_name):
    assert isinstance(data_name, str)
    data_cls = class_utils.import_class(f"data_io.torch_io.{data_name}")
    return data_cls


def import_data(data_name, args):
    assert isinstance(args, argparse.Namespace)
    data_cls = import_data_class(data_name)
    data = data_cls(args)
    return data


def load_and_print_info(data_module_class) -> None:
    """Load EMNISTLines and print info."""
    parser = argparse.ArgumentParser()
    data_module_class.add_to_argparse(parser)
    args = parser.parse_args()
    dataset = data_module_class(args)
    dataset.prepare_data()
    dataset.setup()
    print(dataset)
