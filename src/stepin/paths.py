import os
from pathlib import Path

CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "stepin"
CACHE.mkdir(parents=True, exist_ok=True)
LOG = str(CACHE / "stepin.log")
