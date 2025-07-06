import random
import os
import argparse
import glob
import json

import cv2

from helpers import generate_color

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize data in YOLO format")
    parser.add_argument(
        "--datadir", default="./data/images", type=str, help="Path to images directory"
    )
    parser.add_argument(
        "--classes",
        default="./classes.txt",
        type=str,
        help="Path to classes.txt or class_mapping.json",
    )
    # parser.add_argument('--classes_exclude', nargs='+', type=str, help='List of classes not visualize')
    parser.add_argument(
        "--savename",
        default="./synthesis",
        type=str,
        help="Path to save synthesis images directory",
    )
    parser.add_argument(
        "--number", default=1, type=int, help="Number of visualized labels"
    )
    parser.add_argument("--random", action="store_true", help="Visualize random labels")

    args = parser.parse_args()

    if args.classes.endswith(".txt"):
        with open(args.classes, "r") as f:
            list_classname = f.read().rstrip("\n").split("\n")
    elif args.classes.endswith(".json"):
        with open(args.classes, "r") as f:
            class_mapping_dict = json.load(f)

        swap_dict = {}
        for key in class_mapping_dict.keys():
            swap_dict[class_mapping_dict[key]] = key

        list_classname = []
        for i in range(len(swap_dict.keys())):
            list_classname.append(swap_dict[i])
    else:
        raise NotImplementedError

    class_color_dict = {}
    for class_idx, classname in enumerate(list_classname):
        class_color_dict[class_idx] = {
            "classname": classname,
            "color": generate_color(),
        }

    print(class_color_dict)
    images_dir = os.path.abspath(args.datadir)
    labels_dir = os.path.abspath(os.path.join(images_dir, "..", "labels"))
    list_image_path = glob.glob("./*", root_dir=images_dir)
    assert len(list_image_path) > 0
    if args.random:
        random.shuffle(list_image_path)
    if args.number > 0:
        list_image_path = list_image_path[: args.number]

    os.makedirs(args.savename, exist_ok=True)

    for image_rel_path in list_image_path:
        image_path = os.path.abspath(os.path.join(images_dir, image_rel_path))
        image = cv2.imread(image_path)
        img_h, img_w, _ = image.shape
        label_rel_path = os.path.splitext(image_rel_path)[0] + ".txt"
        label_path = os.path.abspath(os.path.join(labels_dir, label_rel_path))
        with open(label_path, "r") as f:
            list_label = f.read().rstrip("\n").split("\n")

        for label in list_label:
            label_component = label.split(" ")
            class_idx = int(label_component[0])
            classname = class_color_dict[class_idx]["classname"]
            color = class_color_dict[class_idx]["color"]
            bbox_x_center = float(label_component[1]) * img_w
            bbox_y_center = float(label_component[2]) * img_h
            bbox_w = float(label_component[3]) * img_w
            bbox_h = float(label_component[4]) * img_h

            bbox_x1 = int(bbox_x_center - bbox_w / 2.0)
            bbox_y1 = int(bbox_y_center - bbox_h / 2.0)
            bbox_x2 = int(bbox_x_center + bbox_w / 2.0)
            bbox_y2 = int(bbox_y_center + bbox_h / 2.0)

            thickness = 2

            cv2.rectangle(
                image, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), color, thickness
            )

            cv2.putText(
                image,
                classname,
                (bbox_x1, bbox_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

        cv2.imwrite(os.path.join(args.savename, image_rel_path), image)
