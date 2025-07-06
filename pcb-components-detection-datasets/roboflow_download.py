import os
import requests
from roboflow import Roboflow
from tqdm import tqdm
from my_secrets import API_KEY, PROJECT_ID

rf = Roboflow(API_KEY)

project = rf.project(PROJECT_ID)

records = []

for page in project.search_all(
    # prompt="filename:PCBA",
    offset=0,
    limit=100,
    in_dataset=True,
    batch=False,
    fields=["id", "name", "owner"],
):
    records.extend(page)
    print(".")

print(f"{len(records)} total images found")

# records = [record for record in records if str(record["name"]).startswith("PCBA")]
# print(sorted([record["name"] for record in records]))
# print(f"{len(records)} filtered images found")

for record in tqdm(records):
    base_url = "https://source.roboflow.com"
    url = f"{base_url}/{record['owner']}/{record['id']}/original.jpg"
    # record["id"] + "-" +
    save_path = os.path.join("temp_images", record["id"] + "-" + record["name"])
    if not os.path.exists(save_path):
        try:
            response = requests.get(url)
            response.raise_for_status()

            # Save to temp directory
            with open(save_path, "wb") as f:
                f.write(response.content)

            # print(f"Downloaded: {save_path}")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading image: {e}")
    else:
        pass
        # print(f"Skipping: {save_path}")

"""

### PCB defects.v2i.yolov11

From https://universe.roboflow.com/uni-4sdfm/pcb-defects

This dataset helps detect missing components, but it is in Roboflow, which does not allow us to directly download the original images which are of higher quality. If you go to https://universe.roboflow.com/uni-4sdfm/pcb-defects/dataset/2 and try to "Download Dataset" and select YoloV11 or something, the quality is terrible for the images. Also it contains many augmented images, which are not needed, we will do the augmenting ourselves, while training.

So, the steps I took to download this are:
- Go to https://universe.roboflow.com/uni-4sdfm/pcb-defects
- Select Fork Project
- Fill in the API_KEY and PROJECT_ID in my_secrets.py
- Run roboflow_download.py, this saves all images to ./temp_images
- Run roboflow_save_labels.py, this saves all data to ./temp_data, but it is in Roboflow JSON API response format so we have to convert it to the YOLO format manually.

"""

"""
COMP_DETECT_DIR = Path("./CompDetect_data")
if path.exists(COMP_DETECT_DIR):
    shutil.rmtree(COMP_DETECT_DIR)
os.mkdir(COMP_DETECT_DIR)
DATASET_GROUPS = ["train", "test", "valid"]
for g in DATASET_GROUPS:
    os.mkdir(COMP_DETECT_DIR / g)
# Define metadata for dataset
with open(COMP_DETECT_DIR / "data.yaml", "w") as f:
    f.write(
        f""train: ../train/images
val: ../valid/images
test: ../test/images

nc: {len(CLASSES)}
names: [{",".join(f"'{s}'" for s in CLASSES)}]
""
    )
    """
