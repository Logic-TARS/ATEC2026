# Created by skywoodsz on 2026/01/28.

import os

# Support override via environment variable
if "ATEC_ASSETS_MODEL_DIR" in os.environ:
    ATEC_ASSETS_MODEL_DIR = os.environ["ATEC_ASSETS_MODEL_DIR"]
else:
    # Default: external/atec_robot_model relative to repo root
    _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    ATEC_ASSETS_MODEL_DIR = os.path.join(_REPO_ROOT, "external", "atec_robot_model")