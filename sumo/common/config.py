"""Load config/sumo.yaml and set up logging."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "sumo.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the YAML config. SUMO_CONFIG overrides the path; SUMO_FORCE_MOCK=1
    switches every device to mock."""
    path = Path(path or os.environ.get("SUMO_CONFIG") or DEFAULT_CONFIG)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if os.environ.get("SUMO_FORCE_MOCK") == "1":
        cfg["devices"] = {name: "mock" for name in cfg["devices"]}
    return cfg


def setup_logging(cfg: dict[str, Any], node: str) -> logging.Logger:
    """Configure root logging with the node name in every line."""
    logging.basicConfig(
        level=cfg.get("logging", {}).get("level", "INFO"),
        format=f"%(asctime)s.%(msecs)03d {node:<16} %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )
    # bleak and legoeducation are chatty at INFO
    logging.getLogger("bleak").setLevel(logging.WARNING)
    return logging.getLogger(node)
