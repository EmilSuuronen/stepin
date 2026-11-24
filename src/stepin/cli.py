import argparse
import sys
from pathlib import Path

from .paths import LOG, PHOTOS, builtin, scene_path, title_of


def main():
    from .cells import MODES

    parser = argparse.ArgumentParser(prog="stepin", description="Step inside a photo and run shell commands on what you find there.")
    parser.add_argument("photo", nargs="?", default="livingroom", help=f"an image file, or one of: {', '.join(builtin())}")
    parser.add_argument("--rebuild", action="store_true", help="look at the photo again instead of using the saved scene")
    parser.add_argument("--mode", choices=MODES, default=MODES[0], help="how to draw (default: %(default)s)")
    args = parser.parse_args()
    photo = Path(args.photo)
    known = args.photo in builtin()
    if known and args.rebuild:
        parser.error("built-in photos can't be rebuilt")
    if not known and not photo.is_file():
        parser.error(f"{photo}: no such photo")
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        parser.error("stepin needs a terminal")

    from PIL import Image, UnidentifiedImageError

    if not known:
        try:
            Image.open(photo).verify()
        except (UnidentifiedImageError, OSError, SyntaxError):
            parser.error(f"{photo}: not a photo I can read")

    from .app import App
    from .scene import Build, Scene
    from .worker import Worker

    worker = Worker()
    try:
        if known:
            build, title = Build.load(PHOTOS / f"{args.photo}.npz"), args.photo
        else:
            build, title = look_at(photo, worker, args.rebuild), title_of(photo)
        print("  assembling (the very first time takes a minute)...", flush=True)
        worker.request("warm", build.image)
        app = App(Scene(build), worker, title, MODES.index(args.mode))
        app.prepare()
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        worker.close()


def look_at(photo, worker, rebuild):
    from .scene import Build

    path = scene_path(photo)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not rebuild:
        try:
            return Build.load(path)
        except Exception as error:  # damaged or outdated: just make it again
            print(f"saved scene unusable ({error}), rebuilding")
    print(f"stepping into {photo.name} (the first run downloads about 5 GB of models)")
    worker.request("build", str(photo.resolve()), path)
    while True:
        kind, value = worker.poll(1) or (None, None)
        if kind == "progress":
            print(f"  {value}...", flush=True)
        elif kind == "error":
            sys.exit(f"stepin: {value}\n(details in {LOG})")
        elif kind == "done":
            return Build.load(path)


if __name__ == "__main__":
    main()
