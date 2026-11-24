"""Draws the point cloud: every point is a small square, the nearest one wins. Plain Python, compiled by Numba."""

import math

import numba
import numpy as np
from numba import njit, prange

STYLES = ("photo", "hologram", "depth", "objects")


@njit(cache=True, fastmath=True, parallel=True)
def _project(pos, size, drift, tag, show, offset, rot, eye, f, cx, cy, w, h, time, wander, sx, sy, sz, half):
    """World to screen. Points of a half-vanished tag fly apart; points at depth breaks float when seen from aside."""
    for i in prange(len(pos)):
        sz[i] = -1.0
        t = tag[i]
        s = show[t]
        if s <= 0.0:
            continue
        x, y, z = pos[i, 0] + offset[t, 0], pos[i, 1] + offset[t, 1], pos[i, 2] + offset[t, 2]
        # Cheap per-point noise: three bytes of a scrambled index.
        n = (i * 2654435761) & 0xFFFFFFFF
        a, b, c = (n >> 8 & 255) / 127.5 - 1, (n >> 16 & 255) / 127.5 - 1, (n >> 24 & 255) / 255
        shrink = 1.0
        if s < 1.0:
            u = min(max((1.0 - s) * 1.7 - c * 0.7, 0.0), 1.0)
            if u >= 1.0:
                continue
            reach = u * u * 0.12 * z
            x += a * reach
            y += (b - 1.2) * reach
            z += (c - 0.5) * reach
            shrink = 1.0 - u
        d = drift[i] * wander
        if d > 0.0:
            x += d * a * math.sin(time * 1.1 + 6.0 * c)
            y += d * b * math.sin(time * 0.9 + 5.0 * a)
            z += d * (c - 0.5) * math.sin(time * 0.7 + 4.0 * b)
        x -= eye[0]
        y -= eye[1]
        z -= eye[2]
        vz = rot[2, 0] * x + rot[2, 1] * y + rot[2, 2] * z
        if vz < 0.05:
            continue
        k = f / vz
        px = cx + k * (rot[0, 0] * x + rot[0, 1] * y + rot[0, 2] * z)
        py = cy + k * (rot[1, 0] * x + rot[1, 1] * y + rot[1, 2] * z)
        r = min(max(0.5 * k * size[i] * shrink, 0.5), 14.0)
        if px < -r or py < -r or px > w + r or py > h + r:
            continue
        sx[i], sy[i], sz[i], half[i] = px, py, vz, r


@njit(cache=True, fastmath=True, parallel=True)
def _splat(sx, sy, sz, half, zbuf, ibuf, strips):
    h, w = zbuf.shape
    for s in prange(strips):
        top, bottom = h * s // strips, h * (s + 1) // strips
        for y in range(top, bottom):
            for x in range(w):
                zbuf[y, x] = 1e30
                ibuf[y, x] = -1
        for i in range(len(sz)):
            z = sz[i]
            if z < 0.0:
                continue
            r = half[i]
            # Rounding to the nearest pixel edge makes a one-pixel point cover exactly one pixel.
            y0 = max(int(math.floor(sy[i] - r + 0.5)), top)
            y1 = min(int(math.floor(sy[i] + r + 0.5)), bottom)
            if y0 >= y1:
                continue
            x0 = max(int(math.floor(sx[i] - r + 0.5)), 0)
            x1 = min(int(math.floor(sx[i] + r + 0.5)), w)
            for y in range(y0, y1):
                for x in range(x0, x1):
                    if z < zbuf[y, x]:
                        zbuf[y, x] = z
                        ibuf[y, x] = i


@njit(cache=True, fastmath=True, parallel=True)
def _shade(zbuf, ibuf, col, tag, show, glow, palette, style, near, far, time, rgb, tags):
    h, w = zbuf.shape
    for y in prange(h):
        for x in range(w):
            i = ibuf[y, x]
            if i < 0:
                rgb[y, x, 0] = rgb[y, x, 1] = rgb[y, x, 2] = 0.0
                tags[y, x] = -1
                continue
            t = tag[i]
            tags[y, x] = t
            r, g, b = float(col[i, 0]), float(col[i, 1]), float(col[i, 2])
            light = 0.3 * r + 0.59 * g + 0.11 * b
            if style == 1:
                scan = 0.9 + 0.1 * math.sin(y * 0.4 - time * 4.0)
                v = (30.0 + light * 0.95) * scan
                r, g, b = v * 0.22, v * 0.85, v * 1.0
            elif style == 2:
                u = min(max((zbuf[y, x] - near) / (far - near), 0.0), 1.0)
                r = 255 * min(max(1.5 - abs(4 * (1 - u) - 3), 0.0), 1.0)
                g = 255 * min(max(1.5 - abs(4 * (1 - u) - 2), 0.0), 1.0)
                b = 255 * min(max(1.5 - abs(4 * (1 - u) - 1), 0.0), 1.0)
            elif style == 3:
                m = 0.35 + 0.65 * light / 255
                r, g, b = palette[t, 0] * m, palette[t, 1] * m, palette[t, 2] * m
            if show[t] < 1.0:
                k = min((1.0 - show[t]) * 1.5, 1.0)
                r, g, b = r + (255 - r) * k * 0.45, g + (175 - g) * k * 0.45, b + (95 - b) * k * 0.45
            if glow[t] > 0.0:
                k = glow[t]
                # A bright rim where the tag ends, a soft tint inside.
                rim = False
                for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and (ibuf[yy, xx] < 0 or tag[ibuf[yy, xx]] != t):
                        rim = True
                k *= 0.95 if rim else 0.28
                r, g, b = r + (255 - r) * k, g + (190 - g) * k, b + (80 - b) * k
            rgb[y, x, 0], rgb[y, x, 1], rgb[y, x, 2] = r, g, b


def _palette():
    from .scene import MAX_TAGS

    hue = (np.arange(MAX_TAGS) // 3 * 0.618034) % 1
    channel = np.abs(((hue[:, None] + np.array([0, 2 / 3, 1 / 3])) % 1) * 6 - 3) - 1
    palette = (np.clip(channel, 0, 1) * 0.75 + 0.25) * 255
    palette[0] = 70
    return palette.astype(np.float32)


class Renderer:
    def __init__(self):
        self._palette = None
        self._n = self._shape = None

    def draw(self, scene, camera, width, height, style=0, time=0.0, flat_show=None):
        """Returns (rgb float32 H,W,3; tag per pixel, -1 for empty). `flat_show` replaces the show table."""
        if self._palette is None:
            self._palette = _palette()
        n = len(scene.pos)
        if self._n != n:
            self._n = n
            self._sx, self._sy, self._sz, self._half = (np.zeros(n, np.float32) for _ in range(4))
        if self._shape != (height, width):
            self._shape = (height, width)
            self._z, self._i = np.zeros((height, width), np.float32), np.zeros((height, width), np.int32)
            self._rgb, self._tags = np.zeros((height, width, 3), np.float32), np.zeros((height, width), np.int32)
        show = scene.show if flat_show is None else flat_show
        rot, eye = camera.matrix().astype(np.float32), camera.eye.astype(np.float32)
        f = camera.focal(scene, width, height)
        wander = float(min(np.linalg.norm(eye) / scene.scale, 1.0) * 0.015 * scene.scale)
        _project(scene.pos, scene.size, scene.drift, scene.tag, show, scene.offset, rot, eye, f, width / 2, height / 2,
                 width, height, time, wander, self._sx, self._sy, self._sz, self._half)
        _splat(self._sx, self._sy, self._sz, self._half, self._z, self._i, numba.get_num_threads())
        _shade(self._z, self._i, scene.col, scene.tag, show, scene.glow, self._palette, style,
               0.3 * scene.scale, 2.2 * scene.scale, time, self._rgb, self._tags)
        return self._rgb, self._tags


class Camera:
    """x right, y down, z into the photo. The photo was taken from the origin, looking along z."""

    def __init__(self):
        self.eye = np.zeros(3)
        self.yaw = self.pitch = 0.0

    def matrix(self) -> np.ndarray:
        cy, sy, cp, sp = math.cos(self.yaw), math.sin(self.yaw), math.cos(self.pitch), math.sin(self.pitch)
        return np.array([[cy, 0, -sy], [sp * sy, cp, sp * cy], [cp * sy, -sp, cp * cy]])

    def focal(self, scene, width, height) -> float:
        """Between showing the whole photo and filling the whole window."""
        h, w = scene.valid.shape
        a, b = width / w, height / h
        return scene.focal * math.sqrt(a * b)

    def aim(self, target):
        """Yaw and pitch that look at a point."""
        x, y, z = np.asarray(target) - self.eye
        return math.atan2(x, z), math.atan2(-y, math.hypot(x, z))

    def to_screen(self, point, scene, width, height):
        v = self.matrix() @ (np.asarray(point) - self.eye)
        if v[2] < 0.05:
            return None
        f = self.focal(scene, width, height)
        return width / 2 + f * v[0] / v[2], height / 2 + f * v[1] / v[2]
