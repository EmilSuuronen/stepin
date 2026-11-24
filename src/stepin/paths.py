import hashlib
import os
import re
from pathlib import Path

CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "stepin"
CACHE.mkdir(parents=True, exist_ok=True)
LOG = str(CACHE / "stepin.log")

PHOTOS = Path(__file__).parent / "photos"  # scenes that come with stepin


def builtin() -> list[str]:
    return sorted(p.stem for p in PHOTOS.glob("*.npz"))


def scene_path(photo: Path) -> Path:
    return CACHE / "scenes" / f"{hashlib.sha256(photo.read_bytes()).hexdigest()[:32]}.npz"


def title_of(photo: Path) -> str:
    return re.sub(r"[^\w.-]+", "-", photo.stem) or "photo"
