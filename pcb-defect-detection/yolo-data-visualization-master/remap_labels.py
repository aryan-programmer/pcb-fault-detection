import argparse
import os
import glob
import json
from tqdm import tqdm

def load_list_classname(classname_file):
    if classname_file.endswith(".txt"):
        with open(classname_file, "r") as f:
            list_classname = f.read().rstrip("\n").split("\n")
    elif classname_file.endswith(".json"):
        with open(classname_file, 'r') as f:
            class_mapping_dict = json.load(f)
        
        swap_dict = {}
        for key in class_mapping_dict.keys():
            swap_dict[class_mapping_dict[key]] = key

        list_classname = []
        for i in range(len(swap_dict.keys())):
            list_classname.append(swap_dict[i])
    else:
        raise NotImplementedError
    
    return list_classname

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize data in YOLO format')
    parser.add_argument('--label_dirs', type=str, nargs='+', required=True, help='Path to images directory')
    parser.add_argument('--old_classes', type=str, required=True, help='Path to classes.txt or class_mapping.json')
    parser.add_argument('--new_classes', type=str, required=True, help='Path to classes.txt or class_mapping.json')
    # parser.add_argument('--classes_exclude', nargs='+', type=str, help='List of classes not visualize')

    args = parser.parse_args()
    
    list_old_classname = load_list_classname(args.old_classes)
    list_new_classname = load_list_classname(args.new_classes)

    for label_dir in args.label_dirs:
        label_dir = label_dir.rstrip("/")
        assert label_dir.endswith("labels"), """label_dir must have "labels" dirname (For example: "./pack1/labels")"""

        new_label_dir = label_dir.replace("/labels", "/labels_new")
        os.makedirs(new_label_dir, exist_ok=True)

        list_label_file = glob.glob(f"{label_dir}/*.txt")
        for label_file in tqdm(list_label_file):
            with open(label_file, "r") as f:
                list_line_label = f.read().rstrip("\n").split("\n")
            
            new_list_line_label = []

            for line_label in list_line_label:
                line_data = line_label.split(" ")
                old_class_idx = line_data[0]

                if old_class_idx == "":
                    continue

                old_class_name = list_old_classname[int(old_class_idx)]
                new_class_idx = list_new_classname.index(old_class_name)
                line_data[0] = str(new_class_idx)

                new_line_label = " ".join(line_data)
                new_list_line_label.append(new_line_label)

            save_str = "\n".join(new_list_line_label)
            with open(label_file.replace(label_dir, new_label_dir), "w") as f:
                f.write(save_str)