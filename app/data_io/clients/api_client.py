import requests
import os

if __name__ == "__main__":
    from pathlib import Path
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    print(str(Path(__file__).resolve().parents[2]))

import data_io.img_io
from data_io import pathutils


def request_predict(url_api, path_img):
    url_request = f"{url_api}/v1/predict"
    img_b64 = data_io.img_io.encode_img_2_b64(path_img)
    json_d = {"image": img_b64}
    r = requests.post(url_request, json=json_d)
    return r


def main():
    url_api = "http://127.0.0.1:8000"
    name_data = pathutils.get_data_proc_dir("iam_paragraphs")
    path_img = name_data / "a01-077.png"
    res = request_predict(url_api, path_img)
    print(res.status_code)
    print(res.json())


if __name__ == "__main__":
    main()
