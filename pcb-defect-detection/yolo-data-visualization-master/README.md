# Visualize and Process YOLO format data

## How to run

### Visualize data

``` bash
pip install opencv-python
python visualize.py --datadir ./data/images/ --classes classes.txt --savename output --number 1 --random
```

### Count labels

``` bash
python count_labels.py --label_dirs /path/to/a/labels /path/to/b/labels --classes /path/to/classes.txt
```

### Remap labels

``` bash
python remap_labels.py --label_dirs /path/to/a/labels /path/to/b/labels --old_classes /path/to/old_classes.txt --new_classes /path/to/new_classes.txt
```

## Demo

| In | Out |
| --- | --- |
| ![](./assets/demo_in.jpg) | ![](./assets/demo_out.jpg) |