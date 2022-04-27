import sys
from pathlib import Path

try:
    dir_my_home = sys._MEIPASS
except:
    dir_my_home = Path(__file__).resolve().parents[1]
finally:
    pass

dir_app = Path(dir_my_home, "app")

dir_data = Path(dir_my_home, "data")
dir_data_raw = Path(dir_data, "raw")
dir_data_proc = Path(dir_data, "processed")
dir_data_dl = Path(dir_data, "downloaded")
dir_data_nltk = Path(dir_data_dl, "nltk")

dir_workspace = Path(dir_my_home, "workspace")
dir_config = Path(dir_workspace, "configs")
dir_log = Path(dir_workspace, "log")
