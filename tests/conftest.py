import numpy as np
import pytest

from stepin.app import App
from stepin.scene import Build, Scene, rays

H, W, F = 96, 128, 100.0


def make_build():
    """A wall 4 m away with a red box 2 m away and a green button on the box."""
    z = np.full((H, W), 4.0, np.float32)
    image = np.full((H, W, 3), 90, np.uint8)
    box, button = np.zeros((H, W), bool), np.zeros((H, W), bool)
    box[30:70, 40:80] = True
    button[45:55, 55:65] = True
    z[box], image[box] = 2.0, (220, 20, 20)
    image[button] = (20, 220, 20)
    ray = rays((H, W), F)
    plate = ray * 4.0
    return Build(image, ray * z[..., None], np.ones((H, W), bool), F, "a box on a wall", ["button", "box"],
                 np.stack([button, box]), np.full_like(image, 90), plate, np.ones((H, W), bool))


class FakeWorker:
    """Answers immediately, the way the real AI process would."""

    def __init__(self):
        self.busy, self.alive, self._answer, self.found = None, True, None, []

    def request(self, op, *args):
        self.busy = op
        if op == "fill":
            image, points, valid, mask, focal = args
            self._answer = ("done", (np.full_like(image, 90), rays(valid.shape, focal) * 4.0, np.ones_like(valid)))
        elif op == "find":
            self._answer = ("done", self.found)

    def poll(self, timeout=0):
        answer, self._answer = self._answer, None
        if answer:
            self.busy = None
        return answer


@pytest.fixture
def build():
    return make_build()


@pytest.fixture
def scene(build):
    return Scene(build)


@pytest.fixture
def app(scene):
    return App(scene, FakeWorker(), "room")
