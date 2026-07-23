# highlight.py
import argparse
import numpy as np
import torch
import cv2
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import Sam2Model, Sam2Processor

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--object", default="crosswalk")
    p.add_argument("--color", default="0,255,255")  # R,G,B — default yellow
    p.add_argument("--alpha", type=float, default=0.45)
    p.add_argument("--out", default="highlighted.png")
    p.add_argument("--box-threshold", type=float, default=0.3)
    p.add_argument("--text-threshold", type=float, default=0.25)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = Image.open(args.image).convert("RGB")

    # --- Grounding DINO: text prompt -> bounding box ---
    gd_id = "model/grounding-dino-base"
    gd_proc = AutoProcessor.from_pretrained(gd_id, local_files_only=True)
    gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(gd_id, local_files_only=True).to(device)

    inputs = gd_proc(images=image, text=[[args.object]], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = gd_model(**inputs)

    results = gd_proc.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    if len(results["boxes"]) == 0:
        print(f"No '{args.object}' found. Try lowering --box-threshold.")
        return
    boxes = results["boxes"].cpu().numpy()
    print(f"Found {len(boxes)} box(es) for '{args.object}'")

    # --- SAM2: box -> pixel mask ---
    sam_id = "model/sam2.1-hiera-large"
    sam_proc = Sam2Processor.from_pretrained(sam_id, local_files_only=True)
    sam_model = Sam2Model.from_pretrained(sam_id, local_files_only=True).to(device)

    combined_mask = np.zeros((image.height, image.width), dtype=bool)
    for box in boxes:
        sam_inputs = sam_proc(image, input_boxes=[[box.tolist()]], return_tensors="pt").to(device)
        with torch.no_grad():
            sam_out = sam_model(**sam_inputs)
        masks = sam_proc.post_process_masks(
            sam_out.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
        )
        best = masks[0][0][sam_out.iou_scores.argmax()].numpy().astype(bool)
        combined_mask |= best

    # --- Composite the transparent color overlay ---
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    r, g, b = (int(c) for c in args.color.split(","))
    overlay = img_bgr.copy()
    overlay[combined_mask] = (b, g, r)
    out = cv2.addWeighted(overlay, args.alpha, img_bgr, 1 - args.alpha, 0)

    cv2.imwrite(args.out, out)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()