# pcb-fault-detection

## Project Overview

This repository hosts the core machine learning components for an AI-driven optical Printed Circuit Board (PCB) inspection system. Its primary purpose is to develop a computer vision-based inspection tool that significantly reduces inspection time and minimizes costly field returns, a capability especially critical in high-volume EMS (Electronics Manufacturing Services) environments. It includes all necessary code for data acquisition, preprocessing, model training, and evaluation for PCB fault and component detection, as well as artifacts from successful training runs.

This project serves as a companion to the PCB Fault Detection UI, which provides the user interface and is available at [pcb_fault_detection_ui](https://github.com/aryan-programmer/pcb_fault_detection_ui).

## Demo and Application

The `./test_images` directory contains the images used in the demo. You can view the full demonstration on YouTube here: [YouTube Demo Link](https://youtu.be/tCxNRT4C0cI). For detailed instructions on setting up and using the desktop application, please refer to the [`pcb_fault_detection_ui`](https://github.com/aryan-programmer/pcb_fault_detection_ui) repository.

## AI Project Lifecycle & Methodology

![AI Lifecycle](./assets/lifecycle.png)

For a detailed explanation of the methodology and steps followed, please refer to the subsequent sections. Much of this content has been adapted from the Markdown cells within the Jupyter Notebooks used for data pre-processing and model training.

## Step 1: Problem Scoping

**AI-Driven Optical PCB Inspection**

- **Category:** General problem
- **Description:** Manual PCB inspections are time-consuming and error-prone. This project aims to develop a computer vision-based inspection tool to reduce inspection time and avoid costly field returns—especially critical in high-volume EMS environments.

## Step 2: Data Acquisition

We have gathered various types of data from the following sources:

These data sources, detailing defects on copper tracks, were used to train the CopperTrack model.

- [DsPCBSD+](https://universe.roboflow.com/pcb-egrla/dspcbsd/dataset/1)
- [MIXED PCB DEFECT DETECTION](https://data.mendeley.com/datasets/fj4krvmrr5/1)
- [PCB_DATASET](https://www.kaggle.com/datasets/akhatova/pcb-defects)

These datasets contain annotated images of PCBs highlighting actual components. This data can be used to identify missing components by comparing an image with the output of a known good PCB.

- [pcb-oriented-detection](https://www.kaggle.com/datasets/yuyi1005/pcb-oriented-detection)
- [WACV pcb-component-detection](https://sites.google.com/view/chiawen-kuo/home/pcb-component-detection)
- [FICS-PCB](https://www.researchgate.net/publication/344475848_FICS-PCB_A_Multi-Modal_Image_Dataset_for_Automated_Printed_Circuit_Board_Visual_Inspection)
  - Also at [TRUST-HUB](https://trust-hub.org/#/data/fics-pcb)
  - And [Roboflow](https://universe.roboflow.com/erl-n2gvo/component-detection-caevk/browse)
- [PCB-Vision](https://arxiv.org/pdf/2401.06528) ([Zenodo Download](https://zenodo.org/records/10617721))
- [CompDetect Dataset](https://universe.roboflow.com/dataset-lmrsw/compdetect)

This dataset contains annotated images of PCBs with missing soldered components. It was used in the final YouTube demo but not for model training.

- [PCB defects.v2i.yolov11](https://universe.roboflow.com/uni-4sdfm/pcb-defects/dataset/2)

## Step 3: Data Preprocessing & Exploration

### For the Copper Track Defect Detection Model

#### [DsPCBSD+ Dataset](https://www.nature.com/articles/s41597-024-03656-8)

This is the most comprehensive of all datasets, as it contains the most classes. Therefore, its class names were used for the entire aggregated dataset.

Due to inconsistent class naming across datasets, the following mappings (from the DsPCBSD+ Dataset) were applied to standardize labels:

```python
DSCPBSD_MAP = {
		0: SHORT,
		1: SPUR,
		2: SPURIOUS_COPPER,
		3: OPEN,
		4: MOUSE_BITE,
		5: HOLE_BREAKOUT,
		6: SCRATCH,
		7: CONDUCTOR_FOREIGN_OBJECT,
		8: BASE_MATERIAL_FOREIGN_OBJECT,
}
```

#### MIXED PCB DEFECT DATASET

From: [Mendeley Data](https://data.mendeley.com/datasets/fj4krvmrr5/2)

It contains similar data but is labeled differently, thus necessitating label mapping.

```python
MIXED_PCB_DEFECT_DATASET_MAPPING = {
		0: MISSING_HOLE,
		1: MOUSE_BITE,
		2: OPEN,
		3: SHORT,
		4: SPUR,
		5: SPURIOUS_COPPER,
}
```

#### PCB_DATASET

From [Kaggle](https://www.kaggle.com/datasets/akhatova/pcb-defects) by The Open Lab on Human Robot Interaction of Peking University.

It is in Pascal VOC format and thus required conversion to YOLO format.

```python
PCB_DATASET_MAPPING = {
		"missing_hole": MISSING_HOLE,
		"mouse_bite": MOUSE_BITE,
		"spurious_copper": SPURIOUS_COPPER,
		"short": SHORT,
		"spur": SPUR,
		"open_circuit": OPEN,
}
```

#### Sample Images

![Copper Track Defects Final Dataset Sample Images](./assets/copper-track-dataset-sample-output.png)

### For the Component Detection model

Class names to integer mapping:

```python
@enum.verify(enum.UNIQUE, enum.CONTINUOUS)
class Component(enum.IntEnum):
		battery = 0
		button = 1
		buzzer = 2
		capacitor = 3
		clock = 4
		connector = 5
		diode = 6
		display = 7
		fuse = 8
		heatsink = 9
		ic = 10
		inductor = 11
		led = 12
		pads = 13
		pins = 14
		potentiometer = 15
		relay = 16
		resistor = 17
		switch = 18
		transducer = 19
		transformer = 20
		transistor = 21
```

#### CompDetect Dataset

From: [Roboflow Universe](https://universe.roboflow.com/dataset-lmrsw/compdetect).

This dataset is useful for component detection; however, Roboflow does not allow direct download of the original, higher-quality images. For example, if you go to [Roboflow CompDetect v23](https://universe.roboflow.com/dataset-lmrsw/compdetect/dataset/23) and attempt to "Download Dataset" in formats like YoloV11, the image quality is significantly degraded. Additionally, it contains many augmented images that are not needed, as we perform augmentation ourselves during training.

(This dataset appears to be copied from another source, but this is not specified on the dataset page, and I could not find the original source during my data gathering.)

Consequently, the steps taken to download this dataset are:

- Go to [Roboflow CompDetect](https://universe.roboflow.com/dataset-lmrsw/compdetect)
- Select `Fork Project`
- Fill in the `API_KEY` and `PROJECT_ID` in `my_secrets.py`
- Run `roboflow_download.py`, which saves all images to `./temp_images`
- Run `roboflow_save_labels.py`, which saves all data to `./temp_data`. Note that this data is in Roboflow JSON API response format and requires manual conversion to YOLO format.

#### FICS-PCB

Paper: [FICS-PCB: A Multi-Modal Image Dataset](https://www.researchgate.net/publication/344475848_FICS-PCB_A_Multi-Modal_Image_Dataset_for_Automated_Printed_Circuit_Board_Visual_Inspection)

Download from: [TRUST-HUB](https://trust-hub.org/#/data/fics-pcb)

The provided link leads to a ~79GB dataset, with images stored in individual ZIP files and annotations presented in a challenging-to-parse mix of CSV (for class labels) and JSON (for bounding box positions).

However, during my data gathering, I also discovered the dataset mirrored at [Roboflow Universe](https://universe.roboflow.com/erl-n2gvo/component-detection-caevk/).

Therefore, we employed the same methodology as with the `CompDetect` dataset, placing both datasets' images and labels into `./temp_images` and `./temp_data` respectively, and then parsing them simultaneously into the YOLO format.

Additionally, the `CompDetect` dataset includes some, but not all, images from the `WACV` dataset. Therefore, these extra images required removal. Fortunately, their filenames matched the originals, simplifying the filtering process.

#### PCB-Vision

Paper at [arXiv](https://arxiv.org/pdf/2401.06528)

Download from [Zenodo](https://zenodo.org/records/10617721) (11GB).

This dataset presented unique challenges, necessitating significant preprocessing. As a result, we dedicated a separate Jupyter Notebook, `./pcb-components-detection-datasets/preprocess-pcb-vision.ipynb`, to convert the dataset into a YOLO-compatible format, which is then stored at `./pcb-components-detection-datasets/PCBVisionYolo`.

The original dataset specified classes using a grayscale image mask, where each pixel in the mask was assigned a value indicating the presence of a specific component.

- 0 = Nothing
- 1 = IC (represented below as red)
- 2 = Capacitor (represented below as green)
- 3 = Connectors (e.g., DIMM, GPU PCIe, excluding berg strips or screw terminals) (represented below as blue)

![PCB-Vision HSI Masks](./pcb-components-detection-datasets/assets/training_hsi.png)

![PCB-Vision: Sample Image](./pcb-components-detection-datasets/assets/output1.png)

![PCB-Vision: Mask Corresponding to the above Sample Image](./pcb-components-detection-datasets/assets/output2.png)

Unfortunately, this format is not compatible with YOLO, necessitating conversion.

The `preprocess-pcb-vision.ipynb` file performs the following pre-processing steps:

- **Straighten the images:** Some PCBs are tilted, which is problematic as it can lead to issues with YOLO (e.g., loose bounding boxes), even if it might be acceptable for training. (Furthermore, data augmentation is performed during training.). For this, we use PCB mask files from the dataset which specify which pixels of the image have the PCB. This is implemented by the `get_pcb_rotation` function, which performs the following steps:
  - Takes in a PCB mask as input.
  - Smooths out any irregularities.
  - Finds the largest contour (i.e., the most prominent shape, regardless of its irregularity) in the image to isolate the PCB.
  - Finds the smallest rotated bounding box to determine the bounds of the PCB's contour.
  - Then, `rotate_copy` rotates the image (and component mask) by the angle of this rotated bounding box to straighten the PCB.
- From the component mask, for each component type, all separate component contours are found, and then all bounding boxes that fit those contours are derived. These are then saved as labels for YOLO.
- The images are often very dark; therefore, basic color correction is performed by clipping the values in all color channels to be less than their 97.5th percentile.

Final result for this dataset:

![PCB-Vision Sample Images After Conversion to YOLO Format](./assets/pcb-vision-post-processed-output.png)

#### WACV: pcb-component-detection

From [Chia-Wen Kuo's Research Page](https://sites.google.com/view/chiawen-kuo/home/pcb-component-detection).

This dataset is in the Pascal VOC format, requiring conversion to YOLO format.

#### pcb-oriented-detection

From: [Kaggle](https://www.kaggle.com/datasets/yuyi1005/pcb-oriented-detection).

It contains oriented bounding boxes, which necessitated conversion to regular bounding boxes.

The file performs the following preprocessing steps:

- Finds the tightest regular bounding boxes that fit the oriented bounding boxes and saves them.
- Here, the images are also dark; consequently, the same basic color correction as before is performed: clipping the values in all color channels to be less than their 97.5th percentile.

It contains many more classes, often redundant, necessitating a comprehensive class mapping.

#### Tiling

Many datasets contain very tiny bounding boxes for small SMD components. To facilitate better model training by making these components comparatively larger, these images were split into larger tiles, and their bounding box annotations were adjusted accordingly.

This process is analogous to [Slicing Aided Hyper Inference (SAHI)](https://ieeexplore.ieee.org/document/9897990); however, SAHI is exclusively used for inference during deployment, not training. Therefore, this process is referred to as Tiling to differentiate it from Slicing, as Tiling is performed during training. While this concept is likely not new, limited existing code implementations were found online.

It works as follows:

- It reads an image and its corresponding bounding boxes.
- Identifies the smallest dimensions among all bounding boxes within an image.
- Determines the necessary tile size to ensure that the smallest bounding box occupies at least 3% of the tile's area.
- Crops the image into tiles, ensuring at least 10% overlap between them and redistributing the overlap to minimize clipping at the edges.

#### Sample images from consolidated & cropped dataset

![Component Detection Final Dataset Sample Images](./assets/final-components-dataset-output.png)

## Step 4: Modeling

Here, we trained two YOLOv11 nano models, one for each dataset/task, using the Ultralytics library. After iterative training, evaluation, and fine-tuning of model parameters and datasets, the final models were developed.

## Step 5: Evaluation

### Track Defect Detection Model

Here, each model was evaluated in four distinct ways:

- on the entire dataset
- on the entire dataset, treated as a single class (i.e., by combining all problem types into one).
- on the large PCB images dataset
- on the large PCB images dataset, with a single class

#### Evaluation Metrics used

- **box-p (Precision):** The proportion of correct positive predictions out of all positive predictions made by the model.
- **box-r (Recall):** The proportion of actual positive instances that were correctly identified by the model.
- **box-f1 (F1-score):** The harmonic mean of precision and recall, balancing both metrics.
- **box-map (Mean Average Precision - IoU threshold range 0.5 to 0.95):** The average of Average Precisions calculated across multiple IoU thresholds (0.5 to 0.95) and all classes.
- **box-map50 (Mean Average Precision - IoU threshold 0.5):** The Mean Average Precision specifically calculated when a detected box is considered correct if its IoU with a ground truth box is $\ge 0.5$.
- **box-map75 (Mean Average Precision - IoU threshold 0.75):** The Mean Average Precision specifically calculated when a detected box is considered correct if its IoU with a ground truth box is $\ge 0.75$.

The best model has the following evaluation output:

<div>

<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
<th></th>
<th></th>
<th>box-p</th>
<th>box-r</th>
<th>box-f1</th>
<th>box-map</th>
<th>box-map50</th>
<th>box-map75</th>
</tr>
<tr>
<th></th>
<th>class_name</th>
<th></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th></th>
</tr>
</thead>
<tbody>
<tr>
<th rowspan="10" valign="top">Full</th>
<th>Short</th>
<td>0.75861</td>
<td>0.63978</td>
<td>0.69415</td>
<td>0.34406</td>
<td>0.67146</td>
<td>0.29448</td>
</tr>
<tr>
<th>Spur</th>
<td>0.76153</td>
<td>0.53222</td>
<td>0.62655</td>
<td>0.34406</td>
<td>0.67146</td>
<td>0.29448</td>
</tr>
<tr>
<th>Spurious copper</th>
<td>0.61793</td>
<td>0.64026</td>
<td>0.62890</td>
<td>0.34406</td>
<td>0.67146</td>
<td>0.29448</td>
</tr>
<tr>
<th>Open</th>
<td>0.70308</td>
<td>0.70270</td>
<td>0.70289</td>
<td>0.34406</td>
<td>0.67146</td>
<td>0.29448</td>
</tr>
<tr>
<th>Mouse bite</th>
<td>0.69113</td>
<td>0.55098</td>
<td>0.61315</td>
<td>0.34406</td>
<td>0.67146</td>
<td>0.29448</td>
</tr>
<tr>
<th>Hole breakout</th>
<td>0.81272</td>
<td>0.96760</td>
<td>0.88342</td>
<td>0.34406</td>
<td>0.67146</td>
<td>0.29448</td>
</tr>
<tr>
<th>Conductor scratch</th>
<td>0.57191</td>
<td>0.55369</td>
<td>0.56265</td>
<td>0.34406</td>
<td>0.67146</td>
<td>0.29448</td>
</tr>
<tr>
<th>Conductor foreign object</th>
<td>0.58090</td>
<td>0.48402</td>
<td>0.52805</td>
<td>0.34406</td>
<td>0.67146</td>
<td>0.29448</td>
</tr>
<tr>
<th>Base material foreign object</th>
<td>0.67865</td>
<td>0.75600</td>
<td>0.71524</td>
<td>0.34406</td>
<td>0.67146</td>
<td>0.29448</td>
</tr>
<tr>
<th>Missing hole</th>
<td>0.73747</td>
<td>0.41842</td>
<td>0.53391</td>
<td>0.34406</td>
<td>0.67146</td>
<td>0.29448</td>
</tr>
<tr>
<th>Full (Single Class)</th>
<th>Combined</th>
<td>0.77179</td>
<td>0.65834</td>
<td>0.71057</td>
<td>0.37722</td>
<td>0.74098</td>
<td>0.32281</td>
</tr>
<tr>
<th rowspan="6" valign="top">Large PCBs</th>
<th>Short</th>
<td>0.43373</td>
<td>0.35714</td>
<td>0.39173</td>
<td>0.17780</td>
<td>0.39143</td>
<td>0.11074</td>
</tr>
<tr>
<th>Spur</th>
<td>0.51660</td>
<td>0.23636</td>
<td>0.32433</td>
<td>0.17780</td>
<td>0.39143</td>
<td>0.11074</td>
</tr>
<tr>
<th>Spurious copper</th>
<td>0.47770</td>
<td>0.42857</td>
<td>0.45181</td>
<td>0.17780</td>
<td>0.39143</td>
<td>0.11074</td>
</tr>
<tr>
<th>Open</th>
<td>0.66475</td>
<td>0.43056</td>
<td>0.52262</td>
<td>0.17780</td>
<td>0.39143</td>
<td>0.11074</td>
</tr>
<tr>
<th>Mouse bite</th>
<td>0.50888</td>
<td>0.40000</td>
<td>0.44792</td>
<td>0.17780</td>
<td>0.39143</td>
<td>0.11074</td>
</tr>
<tr>
<th>Missing hole</th>
<td>0.59376</td>
<td>0.47629</td>
<td>0.52858</td>
<td>0.17780</td>
<td>0.39143</td>
<td>0.11074</td>
</tr>
<tr>
<th>Large PCBs (Single Class)</th>
<th>Combined</th>
<td>0.59917</td>
<td>0.42411</td>
<td>0.49666</td>
<td>0.19223</td>
<td>0.43753</td>
<td>0.11418</td>
</tr>
</tbody>
</table>

</div>

The best model demonstrates strong performance on the full dataset, with only a slight decrease in performance on the large PCB dataset (as indicated by its F1-vs-Confidence curve).

#### Confusion Matrix

![Track Defect Detection Model Confusion Matrix](./pcb-defect-detection/yolo11n_general_2/val_full_dataset/confusion_matrix.png)

#### Normalized Confusion Matrix

![Track Defect Detection Model Normalized Confusion Matrix](./pcb-defect-detection/yolo11n_general_2/val_full_dataset/confusion_matrix_normalized.png)

#### Precision-vs-Confidence Curve

![Track Defect Detection Model Precision-vs-Confidence Curve](./pcb-defect-detection/yolo11n_general_2/val_full_dataset/P_curve.png)

#### Recall-vs-Confidence Curve

![Track Defect Detection Model Recall-vs-Confidence Curve](./pcb-defect-detection/yolo11n_general_2/val_full_dataset/R_curve.png)

#### F1-vs-Confidence Curve

![Track Defect Detection Model F1-vs-Confidence Curve](./pcb-defect-detection/yolo11n_general_2/val_full_dataset/F1_curve.png)

#### Precision-vs-Recall Curve

![Track Defect Detection Model Precision-vs-Recall Curve](./pcb-defect-detection/yolo11n_general_2/val_full_dataset/PR_curve.png)

#### Results

The model performs commendably. Its F1-vs-Confidence curve on the general dataset is notably wide and high, indicating robust performance across a broad range of confidence values, or probability thresholds. Its confusion matrix shows high values, though it occasionally produces false positives, particularly evident in the "background" row.

The optimal probability threshold for this model is 0.25 (25%), which will be used for final deployment.

Thus, this is designated as the final model for deployment: **Model CopperTrack**

### Component Detection Model

Here, we evaluate the model in two ways:

- on the entire dataset
- on the entire dataset, with a single class (by combining all component types as the same)

The best results are as follows:

<div>

<table border="1" class="dataframe">
<thead>
<tr style="text-align: right;">
<th></th>
<th></th>
<th>box-p</th>
<th>box-r</th>
<th>box-f1</th>
<th>box-map</th>
<th>box-map50</th>
<th>box-map75</th>
</tr>
<tr>
<th></th>
<th>class_name</th>
<th></th>
<th></th>
<th></th>
<th></th>
<th></th>
<th></th>
</tr>
</thead>
<tbody>
<tr>
<th rowspan="19" valign="top">Full</th>
<th>battery</th>
<td>0.48056</td>
<td>0.71429</td>
<td>0.57457</td>
<td>0.38478</td>
<td>0.61576</td>
<td>0.41728</td>
</tr>
<tr>
<th>button</th>
<td>0.88862</td>
<td>0.36275</td>
<td>0.51519</td>
<td>0.38478</td>
<td>0.61576</td>
<td>0.41728</td>
</tr>
<tr>
<th>buzzer</th>
<td>0.91834</td>
<td>0.84615</td>
<td>0.88077</td>
<td>0.38478</td>
<td>0.61576</td>
<td>0.41728</td>
</tr>
<tr>
<th>capacitor</th>
<td>0.85284</td>
<td>0.62589</td>
<td>0.72195</td>
<td>0.38478</td>
<td>0.61576</td>
<td>0.41728</td>
</tr>
<tr>
<th>clock</th>
<td>0.60928</td>
<td>0.41667</td>
<td>0.49489</td>
<td>0.38478</td>
<td>0.61576</td>
<td>0.41728</td>
</tr>
<tr>
<th>connector</th>
<td>0.60877</td>
<td>0.48309</td>
<td>0.53870</td>
<td>0.38478</td>
<td>0.61576</td>
<td>0.41728</td>
</tr>
<tr>
<th>diode</th>
<td>0.68364</td>
<td>0.55175</td>
<td>0.61066</td>
<td>0.38478</td>
<td>0.61576</td>
<td>0.41728</td>
</tr>
<tr>
<th>display</th>
<td>0.44909</td>
<td>0.70000</td>
<td>0.54715</td>
<td>0.38478</td>
<td>0.61576</td>
<td>0.41728</td>
</tr>
<tr>
<th>fuse</th>
<td>0.59291</td>
<td>0.80000</td>
<td>0.68106</td>
<td>0.38478</td>
<td>0.61576</td>
<td>0.41728</td>
</tr>
<tr>
<th>ic</th>
<td>0.71568</td>
<td>0.83037</td>
<td>0.76877</td>
<td>0.38478</td>
<td>0.61576</td>
<td>0.41728</td>
</tr>
<tr>
<th>inductor</th>
<td>0.47205</td>
<td>0.39218</td>
<td>0.42842</td>
<td>0.38478</td>
<td>0.61576</td>
<td>0.41728</td>
</tr>
<tr>
<th>led</th>
<td>0.68681</td>
<td>0.49583</td>
<td>0.57590</td>
<td>0.38478</td>
<td>0.61576</td>
<td>0.41728</td>
</tr>
<tr>
<th>pads</th>
<td>0.31419</td>
<td>0.05019</td>
<td>0.08656</td>
<td>0.38478</td>
<td>0.61576</td>
<td>0.41728</td>
</tr>
<tr>
<th>pins</th>
<td>0.21008</td>
<td>0.31959</td>
<td>0.25352</td>
<td>0.38478</td>
<td>0.61576</td>
<td>0.41728</td>
</tr>
<tr>
<th>potentiometer</th>
<td>0.71059</td>
<td>0.40948</td>
<td>0.51956</td>
<td>0.38478</td>
<td>0.61576</td>
<td>0.41728</td>
</tr>
<tr>
<th>relay</th>
<td>0.79690</td>
<td>1.00000</td>
<td>0.88697</td>
<td>0.38478</td>
<td>0.61576</td>
<td>0.41728</td>
</tr>
<tr>
<th>resistor</th>
<td>0.86273</td>
<td>0.65675</td>
<td>0.74578</td>
<td>0.38478</td>
<td>0.61576</td>
<td>0.41728</td>
</tr>
<tr>
<th>switch</th>
<td>0.87072</td>
<td>0.68504</td>
<td>0.76680</td>
<td>0.38478</td>
<td>0.61576</td>
<td>0.41728</td>
</tr>
<tr>
<th>transistor</th>
<td>0.75751</td>
<td>0.66376</td>
<td>0.70754</td>
<td>0.38478</td>
<td>0.61576</td>
<td>0.41728</td>
</tr>
<tr>
<th>Full (Single Class)</th>
<th>Combined</th>
<td>0.84495</td>
<td>0.66326</td>
<td>0.74316</td>
<td>0.42653</td>
<td>0.73219</td>
<td>0.44990</td>
</tr>
</tbody>
</table>

</div>

#### Confusion Matrix

![Component Detection Model Confusion Matrix](./pcb-components-detection/yolo11n_thawed/val_full_dataset/confusion_matrix.png)

#### Normalized Confusion Matrix

![Component Detection Model Normalized Confusion Matrix](./pcb-components-detection/yolo11n_thawed/val_full_dataset/confusion_matrix_normalized.png)

#### Precision-vs-Confidence Curve

![Component Detection Model Precision-vs-Confidence Curve](./pcb-components-detection/yolo11n_thawed/val_full_dataset/P_curve.png)

#### Recall-vs-Confidence Curve

![Component Detection Model Recall-vs-Confidence Curve](./pcb-components-detection/yolo11n_thawed/val_full_dataset/R_curve.png)

#### F1-vs-Confidence Curve

![Component Detection Model F1-vs-Confidence Curve](./pcb-components-detection/yolo11n_thawed/val_full_dataset/F1_curve.png)

#### Precision-vs-Recall Curve

![Component Detection Model Precision-vs-Recall Curve](./pcb-components-detection/yolo11n_thawed/val_full_dataset/PR_curve.png)

#### Results

Similarly, this model also demonstrates strong performance. Its F1-vs-Confidence curve on the general dataset is very wide and high, indicating robust performance across a wide range of confidence values, or probability thresholds. Its confusion matrix similarly shows high values, though it occasionally produces false positives, as observed in the "background" row.

The optimal probability threshold for this model is 0.322 (approximately 32%). We will round this down to 30% for final deployment.

## Step 6: Model Deployment

Export the models to ONNX for deployment.

See [`pcb_fault_detection_ui`](https://github.com/aryan-programmer/pcb_fault_detection_ui) for the final deployed desktop application.

Also, view the demo on YouTube here: [YouTube Demo Link](https://youtu.be/tCxNRT4C0cI). The `./test_images` directory contains the images used in the demo.
