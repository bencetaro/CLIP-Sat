from __future__ import annotations

import base64
import io
from functools import lru_cache
from typing import List, Optional, Union

import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.inference.model.model import CLIPClassifier

# ----------------------
# GLOBAL CACHE (important for FastAPI)
# ----------------------
_CLIP_MODEL = None
_CLIP_PROCESSOR = None
_CLIP_CLASSIFIER_CACHE = {}


DEFAULT_CLASS_NAMES: List[str] = [
    "Agriculture",
    "Airport",
    "Beach",
    "Desert",
    "Forest",
    "Grassland",
    "Highway",
    "Lake",
    "Mountain",
    "Parking",
    "Port",
    "Railway",
    "Residential",
    "River",
]


def load_clip_model(model_name: str = "openai/clip-vit-base-patch32"):
    """
    Lazy-load CLIP model + processor once per worker.
    """
    global _CLIP_MODEL, _CLIP_PROCESSOR

    if _CLIP_MODEL is None or _CLIP_PROCESSOR is None:
        _CLIP_MODEL = CLIPModel.from_pretrained(model_name)
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(model_name)

        _CLIP_MODEL.eval()

    return _CLIP_MODEL, _CLIP_PROCESSOR


# ----------------------
# IMAGE LOADING
# ----------------------
def _load_image(image_input: Union[str, bytes, Image.Image]) -> Image.Image:
    """
    Supports:
    - URL
    - file path
    - raw bytes
    - PIL image
    """

    if isinstance(image_input, Image.Image):
        return image_input

    if isinstance(image_input, bytes):
        return Image.open(io.BytesIO(image_input)).convert("RGB")

    if isinstance(image_input, str):
        if image_input.startswith("http"):
            response = requests.get(image_input)
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        else:
            return Image.open(image_input).convert("RGB")

    raise ValueError("Unsupported image format")


def _decode_base64_image(data: str) -> bytes:
    # Allow "data:image/jpeg;base64,...."
    if "," in data and data.strip().lower().startswith("data:"):
        data = data.split(",", 1)[1]
    return base64.b64decode(data)

def get_base64(img_path: str):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ----------------------
# MAIN PREPROCESSING
# ----------------------
def clip_inference_preprocessing(
    image_input: Union[str, bytes, Image.Image],
    text_input: Union[str, List[str]]
):
    """
    Prepares inputs for CLIP inference.
    """

    model, processor = load_clip_model()

    image = _load_image(image_input)

    if isinstance(text_input, str):
        text_input = [text_input]

    inputs = processor(
        text=text_input,
        images=image,
        return_tensors="pt",
        padding=True
    )

    return inputs


def clip_image_preprocessing(
    image_input: Union[str, bytes, Image.Image],
    model_name: str = "openai/clip-vit-base-patch32",
):
    """Prepare `pixel_values` as input."""
    _, processor = load_clip_model(model_name=model_name)
    image = _load_image(image_input)
    return processor(images=image, return_tensors="pt")


def load_clip_classifier(
    state_dict_path: Optional[str],
    class_names: Optional[List[str]] = None,
    model_name: str = "openai/clip-vit-base-patch32",
    device: Optional[str] = None,
):
    """
    Load a CLIPClassifier and optionally apply trained weights from `state_dict_path`.
    Cached per `state_dict_path` so repeated requests don't reload weights.
    """
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES
    cache_key = state_dict_path or "__zero_shot__"
    if cache_key in _CLIP_CLASSIFIER_CACHE:
        return _CLIP_CLASSIFIER_CACHE[cache_key]["model"], class_names

    clip_model, _ = load_clip_model(model_name=model_name)
    classifier = CLIPClassifier(clip_model, num_classes=len(class_names))

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    classifier.to(device)
    classifier.eval()

    if state_dict_path:
        state = torch.load(state_dict_path, map_location=device)
        classifier.load_state_dict(state)

    _CLIP_CLASSIFIER_CACHE[cache_key] = {"model": classifier}
    return classifier, class_names


def clip_predict_classes(
    state_dict_path: Optional[str],
    image_input: Union[str, bytes, Image.Image],
    class_names: Optional[List[str]] = None,
    model_name: str = "openai/clip-vit-base-patch32",
    device: Optional[str] = None,
) -> List[float]:
    """Return probability distribution over `class_names` for one image."""
    model, class_names = load_clip_classifier(
        state_dict_path=state_dict_path,
        class_names=class_names,
        model_name=model_name,
        device=device,
    )

    if device is None:
        device = next(model.parameters()).device.type

    inputs = clip_image_preprocessing(image_input=image_input, model_name=model_name)
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        logits = model(pixel_values)
        probs = logits.softmax(dim=1)
    return probs.squeeze(0).detach().cpu().tolist()


def resolve_image_input(image_url: Optional[str], image_base64: Optional[str]) -> Union[str, bytes]:
    if image_url:
        return image_url
    if image_base64:
        return _decode_base64_image(image_base64)
    raise ValueError("Either image_url or image_base64 must be provided.")


# ----------------------
# INFERENCE
# ----------------------
def clip_predict_similarity(
    image_input,
    text_input
):
    """
    Returns similarity scores between image and text.
    """

    model, _ = load_clip_model()

    inputs = clip_inference_preprocessing(image_input, text_input)

    with torch.no_grad():
        outputs = model(**inputs)

        # logits_per_image: (1, num_texts)
        logits = outputs.logits_per_image

        probs = logits.softmax(dim=1)

    return probs.squeeze().cpu().tolist()
