# highlight-sam3.py
import argparse
import numpy as np
import torch
import cv2
from PIL import Image
from transformers import Sam3Model, Sam3Processor
from draw_utils import auto_color, draw_labeled_boxes, mask_to_box

_proc = None
_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def load_models():
    global _proc, _model
    if _model is None:
        sam3_id = "model/sam3"
        _proc = Sam3Processor.from_pretrained(sam3_id, local_files_only=True)
        _model = Sam3Model.from_pretrained(sam3_id, local_files_only=True).to(_device)
    return _proc, _model


def highlight(image: Image.Image, objects, alpha: float = 0.45,
              threshold: float = 0.3, mask_threshold: float = 0.5, show_boxes: bool = False):
    """Run SAM3 on a PIL image for each object.

    objects: list of {"text": str, "color": (r, g, b)}
    Returns (result_image, message). result_image is None if nothing was found.
    """
    proc, model = load_models()
    image = image.convert("RGB")

    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    overlay = img_bgr.copy()
    messages = []
    box_layers = []
    found_any = False

    for obj in objects:
        object_text, color = obj["text"], obj["color"]

        inputs = proc(images=image, text=object_text, return_tensors="pt").to(_device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = proc.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        if len(results["masks"]) == 0:
            messages.append(f"No '{object_text}' found. Try lowering the threshold.")
            continue
        messages.append(f"Found {len(results['masks'])} instance(s) for '{object_text}'")
        found_any = True

        instance_masks = results["masks"].cpu().numpy().astype(bool)
        combined_mask = instance_masks.any(axis=0)

        r, g, b = (int(c) for c in color)
        bgr_color = (b, g, r)
        overlay[combined_mask] = bgr_color
        if show_boxes:
            boxes = [box for box in (mask_to_box(m) for m in instance_masks) if box is not None]
            box_layers.append((boxes, object_text, bgr_color))

    if not found_any:
        return None, "\n".join(messages)

    out = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)
    for boxes, label, bgr_color in box_layers:
        draw_labeled_boxes(out, boxes, label, bgr_color)

    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return Image.fromarray(out_rgb), "\n".join(messages)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--object", action="append", required=True,
                   help="Object to highlight; repeat --object for multiple objects")
    p.add_argument("--color", action="append", default=[],
                   help="R,G,B for the object at the matching position; auto-assigned if omitted")
    p.add_argument("--alpha", type=float, default=0.45)
    p.add_argument("--out", default="highlighted.png")
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--mask-threshold", type=float, default=0.5)
    p.add_argument("--show-boxes", action="store_true")
    args = p.parse_args()

    image = Image.open(args.image).convert("RGB")
    objects = []
    for i, text in enumerate(args.object):
        if i < len(args.color):
            color = tuple(int(c) for c in args.color[i].split(","))
        else:
            color = auto_color(i)
        objects.append({"text": text, "color": color})

    result, message = highlight(image, objects, args.alpha, args.threshold, args.mask_threshold,
                                 args.show_boxes)
    print(message)
    if result is None:
        return
    result.save(args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
