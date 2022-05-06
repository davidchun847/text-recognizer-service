import requests
import os
import json
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

if __name__ == "__main__":
    from pathlib import Path
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    print(str(Path(__file__).resolve().parents[2]))

from data_io import img_io
from data_io import pathutils


def request_predict(url_api, path_img):
    url_request = f"{url_api}/v1/predict"
    img_b64 = img_io.encode_img_2_b64_urlsafe(path_img)
    json_d = {"image": img_b64}
    json_data = json.dumps(json_d)
    json_dd = json.loads(json_data)
    img_d = img_io.decode_b64_2_img_urlsafe(json_dd["image"], grayscale=True)
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=1)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    r = requests.post(url_request, json=json_data)
    return r


def request_get(url_api):
    url_request = f"{url_api}/"
    r = requests.get(url_request)
    return r



def main():
    url_api = "http://127.0.0.1:8000"
    name_data = pathutils.get_data_proc_dir("iam_paragraphs")
    path_img = name_data / "trainval" / "a01-077.png"
    res = request_get(url_api)
    res = request_predict(url_api, path_img)
    print(res.status_code)
    print(res.json())


if __name__ == "__main__":
    main()
