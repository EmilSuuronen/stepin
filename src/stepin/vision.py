"""The AI models. Everything runs locally; weights download on first use. Only imported by the worker process."""

import hashlib
from functools import cached_property

import numpy as np
import torch
from PIL import Image

from .paths import CACHE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

LAMA_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
LAMA_SHA256 = "7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c"


class Vision:
    """Loads each model the first time it is needed and keeps it warm."""

    @cached_property
    def _moge(self):
        from moge.model.v2 import MoGeModel

        return MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(DEVICE).eval()

    @cached_property
    def _sam(self):
        from transformers import Sam2Model, Sam2Processor

        name = "facebook/sam2.1-hiera-large"
        return Sam2Processor.from_pretrained(name), Sam2Model.from_pretrained(name, dtype=torch.float32).to(DEVICE).eval()

    @cached_property
    def _florence(self):
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        name = "florence-community/Florence-2-large"
        model = Florence2ForConditionalGeneration.from_pretrained(name, dtype=DTYPE).to(DEVICE).eval()
        return AutoProcessor.from_pretrained(name), model

    @cached_property
    def _dino(self):
        from transformers import AutoProcessor, GroundingDinoForObjectDetection

        name = "IDEA-Research/grounding-dino-base"
        return AutoProcessor.from_pretrained(name), GroundingDinoForObjectDetection.from_pretrained(name).to(DEVICE).eval()

    @cached_property
    def _lama(self):
        path = CACHE / "models" / "big-lama.pt"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_suffix(".part")
            torch.hub.download_url_to_file(LAMA_URL, str(part), progress=False)
            part.rename(path)
        # TorchScript can run code, so never load a file that is not the one we know.
        if hashlib.sha256(path.read_bytes()).hexdigest() != LAMA_SHA256:
            path.unlink()
            raise RuntimeError("The LaMa download is corrupt; it was removed, please try again.")
        return torch.jit.load(path, map_location=DEVICE).eval()

    @torch.inference_mode()
    def geometry(self, image: Image.Image):
        """Metric points (H, W, 3; x right, y down, z forward), a validity mask and the focal length in pixels."""
        tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div(255).to(DEVICE)
        out = self._moge.infer(tensor, use_fp16=DEVICE == "cuda")
        points = out["points"].float().cpu().numpy()
        mask = out["mask"].cpu().numpy() & np.isfinite(points).all(-1)
        return points, mask, float(out["intrinsics"][0, 0]) * image.width

    @torch.inference_mode()
    def _florence_task(self, image: Image.Image, task: str):
        processor, model = self._florence
        inputs = processor(text=task, images=image, return_tensors="pt").to(DEVICE, DTYPE)
        ids = model.generate(**inputs, max_new_tokens=1024, num_beams=3, do_sample=False)
        text = processor.batch_decode(ids, skip_special_tokens=False)[0]
        return processor.post_process_generation(text, task=task, image_size=image.size)[task]

    def detect(self, image: Image.Image):
        """Everything Florence can name, as (label, box) pairs."""
        result = self._florence_task(image, "<OD>")
        return list(zip(result["labels"], result["bboxes"]))

    def caption(self, image: Image.Image) -> str:
        return self._florence_task(image, "<DETAILED_CAPTION>").strip()

    @torch.inference_mode()
    def find(self, image: Image.Image, phrases: list[str], threshold: float = 0.3):
        """Grounding DINO: (phrase, score, box) for anything matching the given words."""
        processor, model = self._dino
        inputs = processor(images=image, text=[phrases], return_tensors="pt").to(DEVICE)
        result = processor.post_process_grounded_object_detection(
            model(**inputs), inputs.input_ids, threshold=threshold, text_threshold=0.25, target_sizes=[image.size[::-1]]
        )[0]
        labels = result.get("text_labels", result["labels"])
        return [(l, float(s), b.tolist()) for l, s, b in zip(labels, result["scores"], result["boxes"])]

    @torch.inference_mode()
    def segment(self, image: Image.Image, boxes: list[list[float]]):
        """SAM 2.1: one boolean mask and its predicted quality per box."""
        if not boxes:
            return np.zeros((0, image.height, image.width), bool), np.zeros(0)
        processor, model = self._sam
        inputs = processor(images=image, input_boxes=[boxes], return_tensors="pt").to(DEVICE)
        out = model(**inputs, multimask_output=True)
        full = processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu())[0].numpy().astype(bool)
        quality = out.iou_scores[0].float().cpu().numpy()
        # SAM offers part / whole readings; prefer the one that actually fills the box that was asked for.
        fit = np.zeros_like(quality)
        for i, box in enumerate(boxes):
            for k, mask in enumerate(full[i]):
                ys, xs = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
                if len(ys):
                    x0, y0, x1, y1 = box
                    inter = max(0, min(x1, xs[-1]) - max(x0, xs[0])) * max(0, min(y1, ys[-1]) - max(y0, ys[0]))
                    fit[i, k] = inter / max((x1 - x0) * (y1 - y0) + (xs[-1] - xs[0]) * (ys[-1] - ys[0]) - inter, 1)
        best = (quality + fit).argmax(-1)
        rows = np.arange(len(boxes))
        return full[rows, best], quality[rows, best]

    @torch.inference_mode()
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """LaMa: paint over `mask` with a guess of what is behind it."""
        h, w = mask.shape
        pad = ((0, -h % 8), (0, -w % 8))
        img = np.pad(image, (*pad, (0, 0)), mode="symmetric")
        msk = np.pad(mask, pad, mode="symmetric")
        t = torch.from_numpy(img.copy()).permute(2, 0, 1)[None].float().div(255).to(DEVICE)
        m = torch.from_numpy(msk.copy())[None, None].float().to(DEVICE)
        out = self._lama(t, m)[0].permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()[:h, :w]
        return np.where(mask[..., None], out, image)
