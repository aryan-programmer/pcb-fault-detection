# Function to generate a random color
import glob
import os
from pathlib import Path
import random

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.offsetbox import AnnotationBbox, AuxTransformBox
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
from tqdm import tqdm


def generate_color() -> tuple[float, float, float]:
    # Generate random values for red, green, and blue channels
    r = random.randint(0, 255) / 255
    g = random.randint(0, 255) / 255
    b = random.randint(0, 255) / 255

    # Ensure sufficient contrast for visualization
    if (r + g + b) / 3 < 0.5:
        return generate_color()  # If color is too dark, try again
    else:
        return (r, g, b)


def get_class_colors(n_classes: int) -> dict[int, tuple[float, float, float]]:
    class_color_dict = {}
    for class_idx in range(n_classes):
        class_color_dict[class_idx] = generate_color()
    return class_color_dict


def show_image_with_bounding_box(
    ax: Axes,
    image: np.ndarray | Image.Image,
    classes: list[str],
    class_colors: dict[int, tuple[float, float, float]],
    labels: list[tuple[int, float, float, float, float]],
    show_only_classes: list[int] | None = None,
    box_width: int = 1,
):
    if isinstance(image, Image.Image):
        img_w, img_h = image.size
    else:
        img_h, img_w = image.shape[:2]

    ax.imshow(image)
    ax.axis("off")

    for label in labels:
        class_idx = label[0]
        classname = classes[class_idx]
        color = class_colors[class_idx]
        bbox_x_center = label[1] * img_w
        bbox_y_center = label[2] * img_h
        bbox_w = label[3] * img_w
        bbox_h = label[4] * img_h

        if show_only_classes is not None and class_idx not in show_only_classes:
            continue

        bbox_x1 = int(bbox_x_center - bbox_w / 2.0)
        bbox_y1 = int(bbox_y_center - bbox_h / 2.0)
        # bbox_x2 = int(bbox_x_center + bbox_w / 2.0)
        # bbox_y2 = int(bbox_y_center + bbox_h / 2.0)

        r = Rectangle((bbox_x1, bbox_y1), bbox_w, bbox_h, fc="none", ec=color, lw=1)
        offsetbox = AuxTransformBox(ax.transData)
        offsetbox.add_artist(r)
        ab = AnnotationBbox(
            offsetbox,
            (bbox_x_center, bbox_y_center),
            boxcoords="data",
            pad=0.52,
            fontsize=box_width,
            bboxprops=dict(facecolor="none", edgecolor=color, lw=box_width),
        )
        ax.add_artist(ab)
        l = ax.text(
            bbox_x1,
            bbox_y1,
            classname,
            horizontalalignment="left",
            verticalalignment="bottom",
            color="white",
            fontsize=10,
            weight="bold",
        )


def label_str_to_num(
    v: str,
) -> tuple[int, float, float, float, float]:
    (c, x, y, w, h) = v.split(" ")
    return (int(c), float(x), float(y), float(w), float(h))


def sample_yolo_dataset(
    dir: Path,
    classes: list[str],
    one_from_each_of: list[int] | None = None,
    max_images: int = 18,
    cols: int = 3,
):
    class_colors = get_class_colors(len(classes))

    images_dir = dir / "images"
    labels_dir = dir / "labels"
    list_images = [
        filename
        for ext in ["XCF", "xcf", "jpg", "JPG", "jpeg", "gif", "GIF", "png", "PNG"]
        for filename in glob.glob("*." + ext, root_dir=images_dir)
    ]

    file_with_annotations_list: list[
        tuple[str, list[tuple[int, float, float, float, float]]]
    ] = []

    for image_rel_path in list_images:
        image_path = images_dir / image_rel_path

        label_rel_path = os.path.splitext(image_rel_path)[0] + ".txt"
        label_path = labels_dir / label_rel_path
        with open(label_path, "r") as f:
            annots = [
                label_str_to_num(v)
                for v in f.read().rstrip("\n").split("\n")
                if len(v) != 0
            ]
            file_with_annotations_list.append((image_path, annots))

    classes_set: set[str] | None
    class_to_image_mapping: dict[int,] | None
    class_uniq_mode = False

    if one_from_each_of is not None:
        class_uniq_mode = True
        classes_set = set(one_from_each_of)
        class_to_image_mapping = {v: [] for v in one_from_each_of}

        for file_with_annotations in file_with_annotations_list:
            for annotation in file_with_annotations[1]:
                if annotation[0] in classes_set:
                    class_to_image_mapping[annotation[0]].append(file_with_annotations)
    else:
        class_to_image_mapping = None
        classes_set = None

    # list_images.sort(key=natural_keys)

    if class_uniq_mode:
        max_images = len(one_from_each_of)
    w = 5
    h = 5
    rows = int(np.ceil(max_images / cols))
    plt.figure(figsize=(cols * w, rows * h))

    for i in tqdm(range(0, max_images)):
        class_id = one_from_each_of[i] if class_uniq_mode else None
        if class_uniq_mode and len(class_to_image_mapping[class_id]) == 0:
            continue
        image_path, annots = random.sample(
            (
                class_to_image_mapping[class_id]
                if class_uniq_mode
                else file_with_annotations_list
            ),
            1,
        )[0]

        # image_rel_path = list_images[i]
        ax = plt.subplot(rows, cols, i + 1)
        show_image_with_bounding_box(
            ax,
            Image.open(image_path).convert("RGB"),
            classes,
            class_colors,
            annots,
            [class_id] if class_uniq_mode else None,
        )

    plt.tight_layout()
    plt.show()
