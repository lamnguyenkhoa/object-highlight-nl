# highlight.py
import argparse
import numpy as np
import torch
import cv2
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import Sam2Model, Sam2Processor
from draw_utils import draw_labeled_boxes

_gd_proc = None
_gd_model = None
_sam_proc = None
_sam_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def load_models():
    global _gd_proc, _gd_model, _sam_proc, _sam_model
    if _gd_model is None:
        gd_id = "model/grounding-dino-base"
        _gd_proc = AutoProcessor.from_pretrained(gd_id, local_files_only=True)
        _gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(gd_id, local_files_only=True).to(_device)
    if _sam_model is None:
        sam_id = "model/sam2.1-hiera-large"
        _sam_proc = Sam2Processor.from_pretrained(sam_id, local_files_only=True)
        _sam_model = Sam2Model.from_pretrained(sam_id, local_files_only=True).to(_device)
    return _gd_proc, _gd_model, _sam_proc, _sam_model


def highlight(image: Image.Image, object_text: str, color=(0, 255, 255), alpha: float = 0.45,
              box_threshold: float = 0.3, text_threshold: float = 0.25, show_boxes: bool = False):
    """Run GroundingDINO + SAM2 on a PIL image and return (result_image, message)."""
    gd_proc, gd_model, sam_proc, sam_model = load_models()
    image = image.convert("RGB")

    inputs = gd_proc(images=image, text=[[object_text]], return_tensors="pt").to(_device)
    with torch.no_grad():
        outputs = gd_model(**inputs)

    results = gd_proc.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    if len(results["boxes"]) == 0:
        return None, f"No '{object_text}' found. Try lowering the box threshold."
    boxes = results["boxes"].cpu().numpy()
    message = f"Found {len(boxes)} box(es) for '{object_text}'"

    combined_mask = np.zeros((image.height, image.width), dtype=bool)
    for box in boxes:
        sam_inputs = sam_proc(image, input_boxes=[[box.tolist()]], return_tensors="pt").to(_device)
        with torch.no_grad():
            sam_out = sam_model(**sam_inputs)
        masks = sam_proc.post_process_masks(
            sam_out.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
        )
        best = masks[0][0][sam_out.iou_scores.argmax()].numpy().astype(bool)
        combined_mask |= best

    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    r, g, b = (int(c) for c in color)
    bgr_color = (b, g, r)
    overlay = img_bgr.copy()
    overlay[combined_mask] = bgr_color
    out = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)

    if show_boxes:
        draw_labeled_boxes(out, boxes, object_text, bgr_color)

    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return Image.fromarray(out_rgb), message


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--object", default="crosswalk")
    p.add_argument("--color", default="0,255,255")  # R,G,B — default yellow
    p.add_argument("--alpha", type=float, default=0.45)
    p.add_argument("--out", default="highlighted.png")
    p.add_argument("--box-threshold", type=float, default=0.3)
    p.add_argument("--text-threshold", type=float, default=0.25)
    p.add_argument("--show-boxes", action="store_true")
    args = p.parse_args()

    image = Image.open(args.image).convert("RGB")
    color = tuple(int(c) for c in args.color.split(","))
    result, message = highlight(image, args.object, color, args.alpha, args.box_threshold, args.text_threshold,
                                 args.show_boxes)
    print(message)
    if result is None:
        return
    result.save(args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
