import argparse
import glob
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize data in YOLO format')
    parser.add_argument('--label_dirs', type=str, nargs='+', required=True, help='Path to images directory')
    parser.add_argument('--classes', type=str, required=True, help='Path to classes.txt or class_mapping.json')
    # parser.add_argument('--classes_exclude', nargs='+', type=str, help='List of classes not visualize')

    args = parser.parse_args()
    
    if args.classes.endswith(".txt"):
        with open(args.classes, "r") as f:
            list_classname = f.read().rstrip("\n").split("\n")
    elif args.classes.endswith(".json"):
        with open(args.classes, 'r') as f:
            class_mapping_dict = json.load(f)
        
        swap_dict = {}
        for key in class_mapping_dict.keys():
            swap_dict[class_mapping_dict[key]] = key

        list_classname = []
        for i in range(len(swap_dict.keys())):
            list_classname.append(swap_dict[i])
    else:
        raise NotImplementedError
    
    count_dict = {x: 0 for x in list_classname}

    for label_dir in args.label_dirs:
        label_dir = label_dir.rstrip("/")
        assert label_dir.endswith("labels"), """label_dir must have "labels" dirname (For example: "./pack1/labels")"""

        list_label_file = glob.glob(f"{label_dir}/*.txt")
        for label_file in list_label_file:
            with open(label_file, "r") as f:
                list_line_label = f.read().rstrip("\n").split("\n")
            
            for line_label in list_line_label:
                class_idx = line_label.split(" ")[0]
                if class_idx == "":
                    continue
                try:
                    count_dict[list_classname[int(class_idx)]] += 1
                except:
                    print(label_file)
                    exit()

    print(count_dict)