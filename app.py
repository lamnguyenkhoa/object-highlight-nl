# app.py
import importlib.util
from pathlib import Path

import gradio as gr
from PIL import ImageColor

import highlight as gd_sam2

_spec = importlib.util.spec_from_file_location("highlight_sam3", Path(__file__).parent / "highlight-sam3.py")
sam3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sam3)


def run(image, object_text, pipeline, color_hex, alpha, box_threshold, text_threshold, mask_threshold, show_boxes):
    if image is None:
        return None, "Upload an image first."
    if not object_text or not object_text.strip():
        return None, "Enter what to highlight."

    color = ImageColor.getrgb(color_hex)

    if pipeline == "GroundingDINO + SAM2":
        result, message = gd_sam2.highlight(
            image, object_text, color=color, alpha=alpha,
            box_threshold=box_threshold, text_threshold=text_threshold, show_boxes=show_boxes,
        )
    else:
        result, message = sam3.highlight(
            image, object_text, color=color, alpha=alpha,
            threshold=box_threshold, mask_threshold=mask_threshold, show_boxes=show_boxes,
        )
    return result, message


def toggle_thresholds(pipeline):
    is_dino = pipeline == "GroundingDINO + SAM2"
    return (
        gr.update(visible=True, label="Box threshold" if is_dino else "Detection threshold"),
        gr.update(visible=is_dino),
        gr.update(visible=not is_dino),
    )


with gr.Blocks(title="Object Highlight") as demo:
    gr.Markdown("# Object Highlight\nUpload an image, describe what to highlight, and pick a pipeline.")

    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil", label="Image")
            object_in = gr.Textbox(label="Object to highlight", placeholder="e.g. crosswalk", value="crosswalk")
            pipeline_in = gr.Radio(
                ["GroundingDINO + SAM2", "SAM3"],
                value="GroundingDINO + SAM2",
                label="Pipeline",
            )
            color_in = gr.ColorPicker(label="Highlight color", value="#00FFFF")
            alpha_in = gr.Slider(0, 1, value=0.45, step=0.05, label="Overlay opacity")
            box_threshold_in = gr.Slider(0, 1, value=0.3, step=0.01, label="Box threshold")
            text_threshold_in = gr.Slider(0, 1, value=0.25, step=0.01, label="Text threshold")
            mask_threshold_in = gr.Slider(0, 1, value=0.5, step=0.01, label="Mask threshold", visible=False)
            show_boxes_in = gr.Checkbox(value=False, label="Draw bounding boxes")
            run_btn = gr.Button("Highlight", variant="primary")
        with gr.Column():
            image_out = gr.Image(type="pil", label="Result")
            status_out = gr.Textbox(label="Status", interactive=False)

    pipeline_in.change(
        toggle_thresholds,
        inputs=pipeline_in,
        outputs=[box_threshold_in, text_threshold_in, mask_threshold_in],
    )
    run_btn.click(
        run,
        inputs=[image_in, object_in, pipeline_in, color_in, alpha_in,
                box_threshold_in, text_threshold_in, mask_threshold_in, show_boxes_in],
        outputs=[image_out, status_out],
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
