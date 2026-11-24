"""Runs the AI in its own process, so the picture stays smooth while it thinks."""

import multiprocessing as mp
import os
import traceback

from .paths import LOG


def _serve(pipe):
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Model libraries print freely; keep that off the screen.
    log = os.open(LOG, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.dup2(log, 1)
    os.dup2(log, 2)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    vision = None
    while True:
        try:
            op, args = pipe.recv()
        except EOFError:
            return
        try:
            if vision is None:
                from .vision import Vision

                vision = Vision()
            pipe.send(("done", OPS[op](vision, pipe, *args)))
        except Exception as error:
            traceback.print_exc()
            pipe.send(("error", f"{type(error).__name__}: {error}"))


def _build(vision, pipe, path, out):
    from .build import build

    build(vision, path, lambda text: pipe.send(("progress", text))).save(out)


def _fill(vision, pipe, image, points, valid, mask, focal):
    from .build import hidden

    return hidden(vision, image, points, valid, mask, focal)


def _find(vision, pipe, image, text):
    from PIL import Image

    from .build import find, iou

    photo = Image.fromarray(image)
    boxes = find(vision, photo, [text], 0.3)[:8]
    masks, quality = vision.segment(photo, [b for _, _, b in boxes])
    kept = []
    for i in range(len(boxes)):
        if quality[i] >= 0.6 and masks[i].any() and all(iou(masks[i], masks[j]) < 0.6 for j in kept):
            kept.append(i)
    return [(boxes[i][1], masks[i]) for i in kept]


def _warm(vision, pipe, image):
    """Run everything once: first runs compile, and nobody should wait for that after typing a command."""
    import numpy as np

    mask = np.zeros(image.shape[:2], bool)
    mask[: len(mask) // 4] = True
    for _ in range(2):
        painted = vision.inpaint(image, mask)
    from PIL import Image

    vision.geometry(Image.fromarray(painted))
    _find(vision, pipe, image, "wall")


OPS = {"build": _build, "fill": _fill, "find": _find, "warm": _warm}


class Worker:
    """One request at a time. Call poll() to receive ("progress" | "done" | "error", value)."""

    def __init__(self):
        context = mp.get_context("spawn")
        self._pipe, child = context.Pipe()
        self._process = context.Process(target=_serve, args=(child,), daemon=True)
        self._process.start()
        child.close()
        self.busy: str | None = None

    def request(self, op: str, *args):
        assert self.busy is None
        self.busy = op
        self._pipe.send((op, args))

    def poll(self, timeout: float = 0):
        if self.busy is None:
            return None
        try:
            if not self._pipe.poll(timeout):
                return None
            message = self._pipe.recv()
        except (EOFError, OSError):
            message = ("error", f"the AI process stopped (see {LOG})")
        if message[0] != "progress":
            self.busy = None
        return message

    @property
    def alive(self) -> bool:
        return self._process.is_alive()

    def close(self):
        self._pipe.close()
        self._process.terminate()
        self._process.join(2)
