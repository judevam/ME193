import os

from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

API_KEY = os.environ["ROBOFLOW_API_KEY"]
WORKSPACE = os.environ["ROBOFLOW_WORKSPACE"]
PROJECT = os.environ["ROBOFLOW_PROJECT"]
VERSION = int(os.environ["ROBOFLOW_VERSION"])


def main():
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    dataset = project.version(VERSION).download("yolov8", location="dataset")
    print(f"Dataset downloaded to: {dataset.location}")


if __name__ == "__main__":
    main()
