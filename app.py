# app.py
import importlib.util
from pathlib import Path

import gradio as gr
from PIL import ImageColor

import highlight as gd_sam2
from draw_utils import auto_color_hex

_spec = importlib.util.spec_from_file_location("highlight_sam3", Path(__file__).parent / "highlight-sam3.py")
sam3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sam3)

MAX_OBJECTS = 6


def run(image, pipeline, alpha, box_threshold, text_threshold, mask_threshold, show_boxes, *texts_and_colors):
    if image is None:
        return None, "Upload an image first."

    texts = texts_and_colors[:MAX_OBJECTS]
    colors = texts_and_colors[MAX_OBJECTS:]
    objects = [
        {"text": text.strip(), "color": ImageColor.getrgb(color)}
        for text, color in zip(texts, colors)
        if text and text.strip()
    ]
    if not objects:
        return None, "Enter at least one object to highlight."

    if pipeline == "GroundingDINO + SAM2":
        result, message = gd_sam2.highlight(
            image, objects, alpha=alpha,
            box_threshold=box_threshold, text_threshold=text_threshold, show_boxes=show_boxes,
        )
    else:
        result, message = sam3.highlight(
            image, objects, alpha=alpha,
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


def row_visibility(count):
    return [gr.update(visible=(i < count)) for i in range(MAX_OBJECTS)]


def add_object(count):
    count = min(count + 1, MAX_OBJECTS)
    return (count, *row_visibility(count))


def remove_object(count):
    removed_index = count - 1 if count > 1 else None
    count = max(count - 1, 1)
    text_updates = [gr.update(value="") if i == removed_index else gr.update() for i in range(MAX_OBJECTS)]
    return (count, *row_visibility(count), *text_updates)


with gr.Blocks(title="Object Highlight") as demo:
    gr.Markdown("# Object Highlight\nUpload an image, list the objects to highlight, and pick a pipeline.")

    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil", label="Image")

            gr.Markdown("**Objects to highlight**")
            count_state = gr.State(1)
            object_rows, text_inputs, color_inputs = [], [], []
            for i in range(MAX_OBJECTS):
                with gr.Row(visible=(i == 0)) as row:
                    text_in = gr.Textbox(label=f"Object {i + 1}", placeholder="e.g. human",
                                          value="human" if i == 0 else "")
                    color_in = gr.ColorPicker(label="Color", value=auto_color_hex(i))
                object_rows.append(row)
                text_inputs.append(text_in)
                color_inputs.append(color_in)
            with gr.Row():
                add_btn = gr.Button("+ Add object")
                remove_btn = gr.Button("- Remove object")

            pipeline_in = gr.Radio(
                ["GroundingDINO + SAM2", "SAM3"],
                value="GroundingDINO + SAM2",
                label="Pipeline",
            )
            alpha_in = gr.Slider(0, 1, value=0.45, step=0.05, label="Overlay opacity")
            box_threshold_in = gr.Slider(0, 1, value=0.3, step=0.01, label="Box threshold")
            text_threshold_in = gr.Slider(0, 1, value=0.25, step=0.01, label="Text threshold")
            mask_threshold_in = gr.Slider(0, 1, value=0.5, step=0.01, label="Mask threshold", visible=False)
            show_boxes_in = gr.Checkbox(value=False, label="Draw bounding boxes")
            run_btn = gr.Button("Highlight", variant="primary")
        with gr.Column():
            image_out = gr.Image(type="pil", label="Result")
            status_out = gr.Textbox(label="Status", interactive=False)

    add_btn.click(add_object, inputs=count_state, outputs=[count_state, *object_rows])
    remove_btn.click(remove_object, inputs=count_state, outputs=[count_state, *object_rows, *text_inputs])

    pipeline_in.change(
        toggle_thresholds,
        inputs=pipeline_in,
        outputs=[box_threshold_in, text_threshold_in, mask_threshold_in],
    )
    run_btn.click(
        run,
        inputs=[image_in, pipeline_in, alpha_in, box_threshold_in, text_threshold_in, mask_threshold_in,
                show_boxes_in, *text_inputs, *color_inputs],
        outputs=[image_out, status_out],
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
