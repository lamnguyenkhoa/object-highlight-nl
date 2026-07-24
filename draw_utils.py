# draw_utils.py
import colorsys
import cv2
import numpy as np


def auto_color(index):
    """Deterministic, visually distinct RGB color for a given object index."""
    hue = (index * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def auto_color_hex(index):
    return "#%02x%02x%02x" % auto_color(index)


def mask_to_box(mask):
    """Return (x0, y0, x1, y1) tightly bounding the True region of a 2D bool mask, or None if empty."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def draw_labeled_boxes(img_bgr, boxes, label, color_bgr):
    """Draw bounding boxes with a text label on a BGR image in place."""
    thickness = max(1, round(min(img_bgr.shape[0], img_bgr.shape[1]) / 300))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(img_bgr.shape[0], img_bgr.shape[1]) / 800)
    text_thickness = max(1, thickness - 1)

    for box in boxes:
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        cv2.rectangle(img_bgr, (x0, y0), (x1, y1), color_bgr, thickness)

        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        label_y0 = max(0, y0 - text_h - baseline)
        cv2.rectangle(img_bgr, (x0, label_y0), (x0 + text_w, label_y0 + text_h + baseline), color_bgr, -1)
        text_color = (0, 0, 0) if sum(color_bgr) > 380 else (255, 255, 255)
        cv2.putText(img_bgr, label, (x0, label_y0 + text_h), font, font_scale, text_color, text_thickness,
                    cv2.LINE_AA)
