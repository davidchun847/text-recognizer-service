"""Tests for web app."""
import os
from pathlib import Path
from unittest import TestCase
import unittest
import base64
from flask import Flask

if __name__ == "__main__":
    from pathlib import Path
    import sys

    dir_app = Path(__file__).resolve().parents[2]
    sys.path.append(str(dir_app))
    print(dir_app)

from servers.api_server import ApiServer
from data_io import pathutils

os.environ["CUDA_VISIBLE_DEVICES"] = ""

SUPPORT_DIRNAME = pathutils.get_data_proc_dir("iam_paragraphs")
FILENAME = SUPPORT_DIRNAME / "a01-077.png"
EXPECTED_PRED = "And, since this is election year in West\nGermany, Dr. Adenauer is in a tough\nspot. Joyce Egginton cables: President\nKennedy at his Washington Press con-\nference admitted he did not know\nwhether America was lagging behind\nRussia in missile power. He said he\nwas waiting for his senior military\naides to come up with the answer on\nFebruary 20."


class TestIntegrations(TestCase):
    def setUp(self):
        self.app = ApiServer.app
        assert isinstance(self.app, Flask)

    def test_index(self):
        response = self.app.get("/")
        assert response.get_data().decode() == "Hello, world!"

    def test_predict(self):
        with open(FILENAME, "rb") as f:
            b64_image = base64.b64encode(f.read())
        response = self.app.post(
            "/v1/predict", json={"image": f"data:image/png;base64,{b64_image.decode()}"}
        )
        json_data = response.get_json()
        self.assertEqual(json_data["pred"], EXPECTED_PRED)


if __name__ == "__main__":
    unittest.main()
