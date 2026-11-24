import argparse
import hashlib
import re
import sys
from pathlib import Path

from .paths import CACHE, LOG


def scene_path(photo: Path) -> Path:
    return CACHE / "scenes" / f"{hashlib.sha256(photo.read_bytes()).hexdigest()[:32]}.npz"


def main():
    from .cells import MODES

    parser = argparse.ArgumentParser(prog="stepin", description="Step inside a photo and run shell commands on what you find there.")
    parser.add_argument("photo", type=Path)
    parser.add_argument("--rebuild", action="store_true", help="look at the photo again instead of using the saved scene")
    parser.add_argument("--mode", choices=MODES, default=MODES[0], help="how to draw (default: %(default)s)")
    args = parser.parse_args()
    if not args.photo.is_file():
        parser.error(f"{args.photo}: no such photo")
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        parser.error("stepin needs a terminal")

    from PIL import Image, UnidentifiedImageError

    try:
        Image.open(args.photo).verify()
    except (UnidentifiedImageError, OSError, SyntaxError):
        parser.error(f"{args.photo}: not a photo I can read")

    from .app import App
    from .scene import Build, Scene
    from .worker import Worker

    worker = Worker()
    try:
        cached = scene_path(args.photo)
        cached.parent.mkdir(parents=True, exist_ok=True)
        build = None
        if cached.exists() and not args.rebuild:
            try:
                build = Build.load(cached)
            except Exception as error:  # damaged or outdated: just make it again
                print(f"saved scene unusable ({error}), rebuilding")
        if build is None:
            print(f"stepping into {args.photo.name} (the first run downloads about 5 GB of models)")
            worker.request("build", str(args.photo.resolve()), cached)
            while True:
                kind, value = worker.poll(1) or (None, None)
                if kind == "progress":
                    print(f"  {value}...", flush=True)
                elif kind == "error":
                    sys.exit(f"stepin: {value}\n(details in {LOG})")
                elif kind == "done":
                    break
            build = Build.load(cached)
        print("  assembling (the very first time takes a minute)...", flush=True)
        worker.request("warm", build.image)
        title = re.sub(r"[^\w.-]+", "-", args.photo.stem) or "photo"
        app = App(Scene(build), worker, title, MODES.index(args.mode))
        app.prepare()
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        worker.close()


if __name__ == "__main__":
    main()
