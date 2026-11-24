"""A photo as a 3D scene: a point cloud split into named, editable objects.

Every point carries a tag. The tag tables (show, offset, glow) decide how all points with that tag are drawn,
so removing, moving or highlighting an object never touches the points themselves.
"""

import json
import re
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

MAX_TAGS = 4096
FORMAT = 1
BACKGROUND = 0
# Things other things stand on rather than belong to.
FLAT = {"rug", "carpet", "floor mat", "curtain", "window", "door"}


@dataclass(eq=False)
class Obj:
    name: str
    label: str
    tag: int
    halo: int  # background pixels hugging the outline; they carry the object's colours, so they leave with it
    plate: int  # the guessed background hidden behind the object
    mask: np.ndarray | None  # its pixels in the photo; None for copies
    parent: "Obj | None" = None
    alive: bool = True
    lifted: bool = False  # its place in the photo has already been filled in

    @property
    def path(self) -> str:
        return f"{self.parent.path}/{self.name}" if self.parent else self.name


@dataclass
class Build:
    """What the AI step produces; this is what gets cached."""

    image: np.ndarray
    points: np.ndarray
    valid: np.ndarray
    focal: float
    caption: str
    labels: list[str]
    masks: np.ndarray
    plate_image: np.ndarray
    plate_points: np.ndarray
    plate_valid: np.ndarray

    def save(self, path):
        meta = json.dumps({"format": FORMAT, "focal": self.focal, "caption": self.caption, "labels": self.labels, "shape": self.masks.shape})
        # A file object, because savez would otherwise append .npz to the temporary name.
        part = path.with_suffix(".part")
        with open(part, "wb") as file:
            np.savez_compressed(
                file,
                meta=np.array(meta),
                image=self.image,
                # Points lie on known rays, so their depth is enough.
                depth=self.points[..., 2],
                valid=self.valid,
                masks=np.packbits(self.masks),
                plate_image=self.plate_image,
                plate_depth=self.plate_points[..., 2],
                plate_valid=self.plate_valid,
            )
        part.replace(path)

    @classmethod
    def load(cls, path) -> "Build":
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            if meta["format"] != FORMAT:
                raise ValueError("made by another version of stepin")
            shape = tuple(meta["shape"])
            masks = np.unpackbits(data["masks"], count=int(np.prod(shape))).reshape(shape).astype(bool)
            ray = rays(shape[1:], meta["focal"])
            points, plate = (ray * data[key][..., None] for key in ("depth", "plate_depth"))
            return cls(data["image"], points, data["valid"], meta["focal"], meta["caption"], meta["labels"],
                       masks, data["plate_image"], plate, data["plate_valid"])


def slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") or "thing"


def depth_edges(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Pixels where the surface breaks. 1/z is linear across any flat surface, so its second difference is the break."""
    disparity = np.where(valid, 1 / np.maximum(points[..., 2], 1e-3), 0)
    edge = np.zeros(valid.shape, bool)
    for axis in (0, 1):
        bend = np.abs(np.roll(disparity, 1, axis) - 2 * disparity + np.roll(disparity, -1, axis))
        edge |= bend * points[..., 2] > 0.035
    edge[[0, -1], :] = edge[:, [0, -1]] = False
    return edge & valid


def snap_edges(points, valid, focal):
    """Pixels on a break are a blend of both sides and would hang in mid-air; give them the nearest solid depth."""
    edges = depth_edges(points, valid)
    solid = valid & ~edges
    if not solid.any():
        return points, edges
    y, x = ndimage.distance_transform_edt(~solid, return_distances=False, return_indices=True)
    z = np.where(edges, points[y, x, 2], points[..., 2])
    return rays(valid.shape, focal) * z[..., None], edges


def point_sizes(points: np.ndarray, focal: float) -> np.ndarray:
    """How big each point must be drawn to touch its neighbours; slanted surfaces need more."""
    pitch = points[..., 2] / focal
    gap = np.zeros(pitch.shape, np.float32)
    for axis in (0, 1):
        after = np.linalg.norm(points - np.roll(points, -1, axis), axis=-1)
        # The nearer of the two neighbours, so a break on one side doesn't inflate the point.
        gap = np.maximum(gap, np.minimum(after, np.roll(after, 1, axis)))
    return np.clip(1.15 * gap, 1.15 * pitch, 3 * pitch).astype(np.float32)


def rays(shape, focal: float) -> np.ndarray:
    h, w = shape
    v, u = np.mgrid[:h, :w].astype(np.float32)
    return np.stack([(u + 0.5 - w / 2) / focal, (v + 0.5 - h / 2) / focal, np.ones_like(u)], -1)


class Scene:
    def __init__(self, build: Build):
        self.image = build.image.copy()
        self.points = build.points.copy()
        self.valid = build.valid.copy()
        self.focal = build.focal
        self.caption = build.caption
        self.objects: dict[str, Obj] = {}
        self._next_tag = 1
        self.show = np.zeros(MAX_TAGS, np.float32)
        self.offset = np.zeros((MAX_TAGS, 3), np.float32)
        self.glow = np.zeros(MAX_TAGS, np.float32)
        self.show[BACKGROUND] = 1
        self.undo_stack: list[dict] = []

        h, w = self.valid.shape
        self.front = np.full((h, w), -1, np.int32)  # pixel -> its point
        self.pos = np.zeros((0, 3), np.float32)
        self.col = np.zeros((0, 3), np.uint8)
        self.size = np.zeros(0, np.float32)
        self.drift = np.zeros(0, np.float32)
        self.tag = np.zeros(0, np.int16)

        self.points, edges = snap_edges(self.points, self.valid, self.focal)
        # Points beside a break get to float a little once you leave the photo's own viewpoint.
        distance = ndimage.distance_transform_edt(~edges)
        self._append_grid(self.valid, self.points, self.image, np.clip(1 - distance / 3, 0, 1), BACKGROUND, front=True)
        plate_points, _ = snap_edges(build.plate_points, build.plate_valid, self.focal)
        plate_size = point_sizes(plate_points, self.focal)
        # shortcut: a hidden layer only just behind the object would flicker through it, so thin gaps stay empty.
        behind = build.plate_valid & (plate_points[..., 2] > self.points[..., 2] * 1.02)
        chunks = []
        for label, mask in sorted(zip(build.labels, build.masks), key=lambda item: -item[1].sum()):
            obj = self.add_object(label, mask)
            m = mask & behind
            chunks.append((plate_points[m], build.plate_image[m], plate_size[m], np.zeros(m.sum()), np.full(m.sum(), obj.plate)))
        if chunks:
            self._append(*(np.concatenate(c) for c in zip(*chunks)))
        self._nest()

        depth = self.points[..., 2][self.valid]
        self.scale = float(np.median(depth)) if depth.size else 1.0

    # building -----------------------------------------------------------------

    def new_tag(self) -> int:
        if self._next_tag >= MAX_TAGS:
            raise ValueError("too many objects in this scene")
        self._next_tag += 1
        return self._next_tag - 1

    def _grown(self, mask: np.ndarray, by: int) -> np.ndarray:
        """binary_dilation, but only around the mask: these are small things in big photos."""
        ys, xs = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
        out = np.zeros_like(mask)
        if len(ys):
            part = np.s_[max(ys[0] - by, 0) : ys[-1] + by + 1, max(xs[0] - by, 0) : xs[-1] + by + 1]
            out[part] = ndimage.binary_dilation(mask[part], iterations=by)
        return out

    def _append(self, pos, col, size, drift, tag) -> np.ndarray:
        start = len(self.pos)
        self.pos = np.concatenate([self.pos, pos.astype(np.float32)])
        self.col = np.concatenate([self.col, col.astype(np.uint8)])
        self.size = np.concatenate([self.size, size.astype(np.float32)])
        self.drift = np.concatenate([self.drift, drift.astype(np.float32)])
        self.tag = np.concatenate([self.tag, np.broadcast_to(np.asarray(tag, np.int16), len(pos))])
        return np.arange(start, len(self.pos), dtype=np.int32)

    def _append_grid(self, keep, points, image, drift, tag, front=False):
        index = self._append(points[keep], image[keep], point_sizes(points, self.focal)[keep], drift[keep], tag)
        if front:
            self.front[keep] = index

    def unique_name(self, label: str) -> str:
        base, n = slug(label), 2
        name = base
        while name in self.objects:
            name, n = f"{base}{n}", n + 1
        return name

    def add_object(self, label: str, mask: np.ndarray, name: str | None = None) -> Obj:
        obj = Obj(self.unique_name(name or label), label, self.new_tag(), self.new_tag(), self.new_tag(), mask.copy())
        self.show[[obj.tag, obj.halo, obj.plate]] = 1
        index = self.front[mask]
        self.tag[index[index >= 0]] = obj.tag
        # A smaller thing in front takes its pixels away from the bigger thing.
        for other in self.objects.values():
            if other.mask is not None and not other.lifted:
                other.mask = other.mask & ~mask
        halo = self._grown(mask, 6) & ~mask
        index = self.front[halo]
        index = index[index >= 0]
        self.tag[index[self.tag[index] == BACKGROUND]] = obj.halo
        self.objects[obj.name] = obj
        return obj

    def _nest(self):
        """A cushion on a sofa lives in sofa/: inside its outline, and at about the same depth."""
        z, shapes = self.points[..., 2], {}
        for obj in self.objects.values():
            solid = obj.mask & self.valid
            if solid.sum() < 20:
                continue
            hull = Image.new("1", solid.shape[::-1])
            yx = np.argwhere(obj.mask)
            try:
                corners = yx[ConvexHull(yx).vertices][:, ::-1]
            except QhullError:  # a perfectly straight line of pixels
                continue
            ImageDraw.Draw(hull).polygon([tuple(map(int, p)) for p in corners], fill=1)
            shapes[obj] = (obj.mask.sum(), np.asarray(hull), np.percentile(z[solid], [2, 50, 98]))
        for child, (area, _, (_, middle, _)) in shapes.items():
            homes = []
            for other, (size, hull, (near, _, far)) in shapes.items():
                # Parents always fill a bigger outline, so this can never form a loop.
                if other.label in FLAT | {child.label} or hull.sum() < 1.5 * shapes[child][1].sum():
                    continue
                if near * 0.9 <= middle <= far * 1.05 and (hull & child.mask).sum() > 0.8 * area:
                    homes.append((hull.sum(), other))
            child.parent = min(homes, key=lambda home: home[0])[1] if homes else None

    # reading ------------------------------------------------------------------

    def children(self, parent: Obj | None) -> list[Obj]:
        return [o for o in self.objects.values() if o.alive and o.parent is parent]

    def family(self, obj: Obj) -> list[Obj]:
        return [obj] + [d for c in self.children(obj) for d in self.family(c)]

    def object_points(self, obj: Obj) -> np.ndarray:
        return self.pos[self.tag == obj.tag] + self.offset[obj.tag]

    def bounds(self, obj: Obj):
        """Robust low and high corners in metres, or None when the object has no usable points."""
        points = np.concatenate([self.object_points(o) for o in self.family(obj)])
        if len(points) < 4:
            return None
        return np.percentile(points, 3, axis=0), np.percentile(points, 97, axis=0)

    # editing ------------------------------------------------------------------

    def fill_mask(self, objs: list[Obj]) -> np.ndarray:
        """The pixels that must be repainted if these objects leave their place."""
        mask = np.zeros(self.valid.shape, bool)
        for obj in objs:
            if obj.mask is not None and not obj.lifted:
                mask |= obj.mask
        return self._grown(mask, 6)

    def lift(self, objs: list[Obj], mask: np.ndarray, fill=None) -> int | None:
        """Vacate the objects' pixels. `fill` is (rgb, points, valid) for the whole photo; returns the new points' tag."""
        # Dilation reaches into neighbours; their pixels stay theirs.
        for obj in self.objects.values():
            if obj.alive and obj.mask is not None and not obj.lifted and obj not in objs:
                mask = mask & ~obj.mask
        tag = None
        if fill is not None:
            rgb, points, valid = fill
            tag = self.new_tag()
            points, _ = snap_edges(points, valid, self.focal)
            self.image[mask], self.points[mask], self.valid[mask] = rgb[mask], points[mask], valid[mask]
            self.front[mask] = -1
            self._append_grid(mask & valid, points, rgb, np.zeros(mask.shape), tag, front=True)
        for obj in objs:
            obj.lifted = True
        return tag

    def copy_object(self, obj: Obj) -> Obj:
        box = self.bounds(obj)
        new = Obj(self.unique_name(obj.name), obj.label, self.new_tag(), self.new_tag(), self.new_tag(), None, obj.parent, lifted=True)
        mine = self.tag == obj.tag
        self._append(self.pos[mine], self.col[mine], self.size[mine], np.zeros(mine.sum()), new.tag)
        self.offset[new.tag] = self.offset[obj.tag] + (np.array([box[1][0] - box[0][0], 0, 0]) * 1.1 if box else 0)
        self.objects[new.name] = new
        return new

    def snapshot(self) -> dict:
        """Everything an edit can change. Points are only ever appended, so a count is enough for them."""
        return {
            "count": len(self.pos), "tag": self.tag.copy(), "front": self.front.copy(), "image": self.image.copy(),
            "points": self.points.copy(), "valid": self.valid.copy(), "show": self.show.copy(), "offset": self.offset.copy(),
            "objects": dict(self.objects),
            "state": {o: (o.alive, o.lifted, o.mask, o.parent) for o in self.objects.values()},
        }

    def restore(self, snap: dict):
        n = snap["count"]
        self.pos, self.col, self.size, self.drift = self.pos[:n], self.col[:n], self.size[:n], self.drift[:n]
        for key in ("tag", "front", "image", "points", "valid", "show", "offset"):
            setattr(self, key, snap[key])
        self.objects = snap["objects"]
        for obj, state in snap["state"].items():
            obj.alive, obj.lifted, obj.mask, obj.parent = state
