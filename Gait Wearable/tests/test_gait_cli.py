"""gait.py status formatting (no board or broker needed)."""
import json

from gait import format_status


def test_format_record_status():
    msg = {"mode": "record", "label": "walking", "seconds": 12.3, "samples": 1230, "lost": 0,
           "rate_hz": 100, "peak_x_g": 1.7, "mag_g": 1.05}
    line = format_status("ME193/Jude/gait", json.dumps(msg).encode())
    assert line.startswith("REC walking")
    assert "LOST" not in line


def test_format_shows_lost_samples():
    msg = {"mode": "record", "label": "running", "seconds": 5.0, "lost": 2}
    assert "LOST 2" in format_status("ME193/Jude/gait", json.dumps(msg).encode())


def test_format_online_status():
    assert format_status("ME193/Jude/gait/status", b"offline") == "wearable is offline"
