import os
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import cv2
from PIL import Image as PILImage

from rvai.types import BoundingBox, Point, Integer, String, Image, Class

MOT_CLASS_MAPPING = [
    "Pedestrian",  # 1
    "Person on vehicle",  # 2
    "Car",  # 3
    "Bicycle",  # 4
    "Motorbike",  # 5
    "Non motorized vehicle",  # 6
    "Static person",  # 7
    "Distractor",  # 8
    "Occluder",  # 9
    "Occluder on the ground",  # 10
    "Occluder full",  # 11
    "Reflection",  # 12
    "Crowd",  # 13
    "Person",  # 14, custom added
]


def img_name_to_frame_no(img_name):
    return int(os.path.splitext(os.path.basename(img_name))[0])


class TrackerDataset:
    def __init__(self,
                 img_folder_path: Path,
                 annotations_path: Path,
                 classes_of_interest: list = None,
                 relabel: dict = None,
                 MOT_type = "MOT2020",
                 ):
        self.img_folder_path = img_folder_path
        self.annotations_path = annotations_path
        self.classes_of_interest = classes_of_interest
        self.relabel = relabel
        self.mot_type=MOT_type

    def _get_annotations_for_img(
        self,
        img_name,
        MOT_type="MOT2020",
        classes_of_interest=None,
        relabel=None,
    ):
        """ Returns a list of rvai annotations for the image at img_path.
        classes_of_interest: a list of the class numbers we're interested in, other bboxes will be discarded.
        relabel: a dict of label numbers, with value which label you want to reassign it to. """
        frame_no = img_name_to_frame_no(self.img_folder_path / img_name)
        all_annotations = []
        for line in open(self.annotations_path, "r").readlines():
            frame, id_no, x, y, w, h, class_ = parse_MOT_annotation_line(
                line, MOT_type=MOT_type
            )
            if frame_no == frame:
                if classes_of_interest:
                    if class_ not in classes_of_interest:
                        continue
                if relabel:
                    if class_ in relabel.keys():
                        class_ = relabel[class_]
                bbox = _to_rvai_bbox(x, y, w, h, class_, id_no, self.img_folder_path / img_name)
                all_annotations.append(bbox)

        return all_annotations

    def visualize(self, idx):
        img, bboxes = self.__getitem__(idx)
        img = np.array(img)

        for box in bboxes:
            x1, y1 = int(box.p1.x), int(box.p1.y)
            x2, y2 = int(box.p2.x), int(box.p2.y)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=4)

        plt.figure()
        plt.imshow(img)

    def __len__(self):
        return len(os.listdir(self.img_folder_path))

    def __getitem__(self, idx):
        sorted_images = sorted(os.listdir(self.img_folder_path))
        selected_img = sorted_images[idx]

        img = Image(PILImage.open(str(self.img_folder_path / selected_img)))
        bboxes = self._get_annotations_for_img(selected_img,
                                               MOT_type=self.mot_type,
                                               classes_of_interest=self.classes_of_interest)

        return img, bboxes


def _to_rvai_bbox(x, y, w, h, class_, id_no, img_id):
    """Create an RVAI BoundingBox from the given input parameters."""

    bbox = BoundingBox(Point(x, y), Point(x + w, y + h))
    bbox.set_attributes(
        original_image_id=String(img_id),
        object_id=Integer(id_no),  # required to train FAIRMOT
    )
    bbox.set_class(
        Class(
            class_uuid=MOT_CLASS_MAPPING[class_ - 1],
            name=MOT_CLASS_MAPPING[class_ - 1],
        )
    )

    return bbox


def parse_MOT_annotation_line(line: str, MOT_type="MOT2020"):
    """Correctly parses the information found in a single line of the MOT csv annotations."""
    # TODO: add other MOT versions here. Different years have different parameters
    if MOT_type == "MOT2020":
        frame_no, id_no, x, y, w, h, conf, class_, vis = line.split(",")

    else:
        raise ValueError(f"Invalid MOT type provided: {MOT_type}")

    return (
        int(frame_no),
        int(id_no),
        int(x),
        int(y),
        int(w),
        int(h),
        int(class_),
    )

