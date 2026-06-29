import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import tv_tensors

from pets.utils.load_data import parse_voc_xml


def draw_boxes(ax, img, boxes, labels, title):
    """Draws image + bounding boxes on a given matplotlib axis."""
    ax.imshow(img)
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box.tolist()
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle(
            (xmin, ymin), width, height, linewidth=2, edgecolor="lime", facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            xmin,
            max(ymin - 5, 0),
            label,
            color="white",
            fontsize=10,
            bbox=dict(facecolor="lime", alpha=0.7, pad=1),
        )
    ax.set_title(title)
    ax.axis("off")


def show_transform_with_boxes(xml_path, img_path, transforms):
    """
    Plots an image and its bounding boxes before and after applying
    a list of torchvision.transforms.v2 transforms.

    Args:
        xml_path: path to Pascal VOC-style XML annotation
        img_path: path to the corresponding image file
        transform_list: list of torchvision.transforms.v2 transform instances
                         e.g. [v2.Resize((224, 224)), v2.RandomHorizontalFlip(p=1.0)]
    """
    # --- Load original image + boxes ---
    boxes, labels, (xml_w, xml_h) = parse_voc_xml(xml_path)
    img = Image.open(img_path).convert("RGB")

    # Sanity check: warn if XML size doesn't match actual image size
    if img.size != (xml_w, xml_h):
        print(f"Warning: XML size {(xml_w, xml_h)} != actual image size {img.size}")

    orig_img = img.copy()
    orig_boxes = boxes.clone()

    # --- Wrap boxes as tv_tensors so v2 transforms handle box math ---
    wrapped_boxes = tv_tensors.BoundingBoxes(
        boxes, format="XYXY", canvas_size=(img.height, img.width)
    )  # type:ignore

    transformed_img, transformed_boxes = transforms(img, wrapped_boxes)

    # transformed_img may be a Tensor or PIL Image depending on the pipeline
    if isinstance(transformed_img, torch.Tensor):
        # Handle normalized float tensors vs unnormalized
        disp_img = transformed_img.clone()
        if disp_img.dtype != torch.uint8:
            disp_img = disp_img.clamp(0, 1)
        disp_img = disp_img.permute(1, 2, 0).numpy()
    else:
        disp_img = transformed_img  # still PIL

    # --- Plot side by side ---
    _, axes = plt.subplots(1, 2, figsize=(12, 6))
    draw_boxes(axes[0], orig_img, orig_boxes, labels, "Original")
    draw_boxes(axes[1], disp_img, transformed_boxes, labels, "Transformed")
    plt.tight_layout()
    plt.show()
