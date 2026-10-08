"""Put the "ping pong/" folder on sys.path so tests can `import game`, `import swing`, etc.
(the folder name has a space, so it can't be imported as a package)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
