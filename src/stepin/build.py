"""Photo in, scene out: geometry, objects, and a guess at what is hidden behind them."""

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

from .scene import Build, rays

LONG_SIDE = 1024
MAX_PER_LABEL = 12
MAX_OBJECTS = 64

# Asked for explicitly because general detectors skip them.
VOCABULARY = [
    ["window", "door", "curtain", "mirror", "picture frame", "shelf", "clock"],
    ["rug", "lamp", "cushion", "plant", "vase", "book", "blanket"],
    ["table", "chair", "sofa", "bed", "cabinet", "desk", "stool"],
    ["monitor", "laptop", "keyboard", "phone", "bottle", "cup", "bag"],
    ["person", "dog", "cat", "bird", "car", "bicycle", "boat"],
    ["tree", "rock", "building", "bridge", "statue", "sign", "bench"],
]
ALIASES = {
    "kitchen & dining room table": "table", "coffee table": "table", "couch": "sofa", "sofa bed": "sofa", "houseplant": "plant",
    "potted plant": "plant", "pillow": "cushion", "picture frame": "picture", "computer monitor": "monitor", "television": "tv",
    "man": "person", "woman": "person", "boy": "person", "girl": "person", "flowerpot": "pot", "studio couch": "sofa",
}
# Parts of things, not things.
SKIP = {"human face", "human hair", "human hand", "human arm", "human leg", "human head", "human eye", "human nose", "human mouth",
        "human ear", "clothing", "footwear", "wheel", "tire", "house", "furniture", "room", "floor", "wall", "ceiling"}


def open_photo(path) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    image.thumbnail((LONG_SIDE, LONG_SIDE), Image.Resampling.LANCZOS)
    return image


def clean_label(label: str) -> str:
    label = label.lower().strip()
    return ALIASES.get(label, label)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return (a & b).sum() / union if union else 0.0


def pick_objects(labels, masks, quality, area):
    """Drop poor, tiny, huge and duplicate masks. Earlier candidates win."""
    kept: list[int] = []
    for i in range(len(labels)):
        share = masks[i].sum() / area
        if quality[i] < 0.7 or not 0.0006 < share < 0.55:
            continue
        if any(iou(masks[i], masks[j]) > 0.6 for j in kept):
            continue
        # The same word around several things at once ("cushion" over the whole row).
        if sum((masks[j] & masks[i]).sum() > 0.8 * masks[j].sum() for j in kept if labels[j] == labels[i]) >= 1:
            continue
        kept.append(i)
    by_size = sorted(kept, key=lambda i: -masks[i].sum())
    out, count = [], {}
    for i in by_size:
        count[labels[i]] = count.get(labels[i], 0) + 1
        if count[labels[i]] <= MAX_PER_LABEL:
            out.append(i)
    return out[:MAX_OBJECTS]


def find(vision, image: Image.Image, phrases, threshold):
    """Grounding DINO boxes, each labelled with exactly one of the phrases."""
    out = []
    for text, score, box in vision.find(image, phrases, threshold):
        label = next((p for p in phrases if p == text), None) or next((p for p in phrases if p in text), None)
        if label:
            out.append((label, score, box))
    return sorted(out, key=lambda item: -item[1])


def detect(vision, image: Image.Image):
    found = [(clean_label(l), b) for l, b in vision.detect(image)]
    for chunk in VOCABULARY:
        found += [(clean_label(l), b) for l, _, b in find(vision, image, chunk, 0.38)]
    # A box around nearly everything is the scene itself ("building"), not a thing in it.
    whole = 0.7 * image.width * image.height
    return [(l, b) for l, b in found if l not in SKIP and (b[2] - b[0]) * (b[3] - b[1]) < whole]


def hidden(vision, image: np.ndarray, points: np.ndarray, valid: np.ndarray, mask: np.ndarray, focal: float):
    """What is behind `mask`: repaint it, measure the repainted photo, and fit that onto the surroundings."""
    painted = vision.inpaint(image, mask)
    guess, guess_valid, _ = vision.geometry(Image.fromarray(painted))
    z = np.where(guess_valid, guess[..., 2], 0)
    ok = valid & guess_valid & ~mask
    reach = max(12, int(np.sqrt(mask.sum()) / 6))
    ring = ok & ndimage.binary_dilation(mask, iterations=reach)
    if ring.sum() < 50:
        return painted, points, np.zeros_like(valid)
    ratio = np.zeros(mask.shape, np.float32)
    ratio[ring] = np.log(points[..., 2][ring] / np.maximum(z[ring], 1e-3))
    middle = np.median(ratio[ring])
    # Ignore ring pixels on something nearer or farther than the surface being continued.
    ring &= np.abs(ratio - middle) < 0.12
    weight = ndimage.gaussian_filter(ring.astype(np.float32), reach)
    smooth = ndimage.gaussian_filter(np.where(ring, ratio - middle, 0), reach) / np.maximum(weight, 1e-4)
    z = z * np.exp(middle + np.where(weight > 1e-4, smooth, 0))
    good = guess_valid & (z > 0)
    return painted, (rays(mask.shape, focal) * np.where(good, z, 0)[..., None]).astype(np.float32), good


def build(vision, path, progress) -> Build:
    photo = open_photo(path)
    image = np.asarray(photo)
    progress("measuring depth")
    points, valid, focal = vision.geometry(photo)
    points = np.where(valid[..., None], points, 0).astype(np.float32)
    if valid.mean() < 0.02:
        raise ValueError("couldn't find any solid surfaces in this photo")
    progress("describing the photo")
    caption = vision.caption(photo)
    progress("naming things")
    found = detect(vision, photo)
    progress(f"cutting out {len(found)} candidates with Segment Anything")
    masks, quality = vision.segment(photo, [b for _, b in found])
    labels = [l for l, _ in found]
    chosen = pick_objects(labels, masks, quality, valid.size)
    labels, masks = [labels[i] for i in chosen], masks[chosen]
    plate = (image, points, np.zeros_like(valid))
    if len(masks):
        progress("imagining what is behind them")
        everything = ndimage.binary_dilation(masks.any(0), iterations=6)
        plate = hidden(vision, image, points, valid, everything, focal)
    return Build(image, points, valid, focal, caption, labels, masks, plate[0], plate[1], plate[2])
