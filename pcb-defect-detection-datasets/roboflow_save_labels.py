import os
from typing import Dict, List, Optional, Union
import requests
from roboflow import Roboflow, Project
from tqdm import tqdm

from my_secrets import API_KEY, PROJECT_ID

rf = Roboflow(API_KEY)

project = rf.project(PROJECT_ID)

records = []


def search(
    project,
    like_image: Optional[str] = None,
    prompt: Optional[str] = None,
    offset: int = 0,
    limit: int = 100,
    tag: Optional[str] = None,
    class_name: Optional[str] = None,
    in_dataset: Optional[str] = None,
    batch: bool = False,
    batch_id: Optional[str] = None,
    fields: Optional[List[str]] = None,
):
    """
    Search for images in a project.

    Args:
        like_image (str): name of an image in your dataset to use if you want to find images similar to that one
        prompt (str): search prompt
        offset (int): offset of results
        limit (int): limit of results
        tag (str): tag that an image must have
        class_name (str): class name that an image must have
        in_dataset (str): dataset that an image must be in
        batch (bool): whether the image must be in a batch
        batch_id (str): batch id that an image must be in
        fields (list): fields to return in results (default: ["id", "created", "name", "labels"])

    Returns:
        A list of images that match the search criteria.

    Example:
        >>> import roboflow

        >>> rf = roboflow.Roboflow(api_key="")

        >>> project = rf.workspace().project("PROJECT_ID")

        >>> results = project.search(query="cat", limit=10)
    """  # noqa: E501 // docs
    if fields is None:
        fields = ["id", "created", "name", "labels"]

    payload: Dict[str, Union[str, int, List[str]]] = {}

    if like_image is not None:
        payload["like_image"] = like_image

    if prompt is not None:
        payload["prompt"] = prompt

    if offset is not None:
        payload["offset"] = offset

    if limit is not None:
        payload["limit"] = limit

    if tag is not None:
        payload["tag"] = tag

    if class_name is not None:
        payload["class_name"] = class_name

    if in_dataset is not None:
        payload["in_dataset"] = in_dataset

    if batch is not None:
        payload["batch"] = batch

    if batch_id is not None:
        payload["batch_id"] = batch_id

    payload["fields"] = fields

    data = requests.post(
        "https://api.roboflow.com"
        + "/"
        + __workspace
        + "/"
        + __project_name
        + "/search?api_key="
        + API_KEY,
        json=payload,
    )

    return data.json()


# print(
#     search(
#         project,
#         offset=0,
#         limit=300,
#         in_dataset=True,
#         batch=False,
#         fields=["id", "name", "owner"],
#     )
# )
# os._exit()

for page in project.search_all(
    offset=0,
    limit=300,
    in_dataset=True,
    batch=False,
    fields=["id", "name", "owner"],
):
    records.extend(page)
    print(".")

print(f"{len(records)} images found")

temp = project.id.rsplit("/")
__workspace = temp[0]
__project_name = temp[1]
for record in tqdm(records):
    save_path = (
        os.path.splitext(
            os.path.join("temp_data", record["id"] + "-" + record["name"])
        )[0]
        + ".json"
    )
    if not os.path.exists(save_path):
        try:
            url = f"https://api.roboflow.com/{__workspace}/{__project_name}/images/{record['id']}?api_key={API_KEY}"
            response = requests.get(url).text

            # Save to temp directory
            with open(save_path, "w") as f:
                f.write(response)

            print(f"Downloaded: {save_path}")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading data: {e}")
    else:
        print(f"Skipping: {save_path}")
