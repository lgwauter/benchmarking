from rvai.cells.deep_sort import DeepSORTCell
from rvai.cells.deep_sort.model import DeepSORTModel
from rvai.cells.yolov5 import Yolov5Cell
from rvai.cells.fairmot import FairMOTCell
from rvai.types import Enum, String
from packing import unpack_model
from pathlib import Path
import torch
import datetime
from typing import List as tList, Optional, Tuple, Sequence
import os
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import numpy as np
import cv2

from rvai.base.context import (
    Context,
    InferenceContext,
    ModelContext,
    TestContext,
    TrainingContext
)

from rvai.types import (
    Boolean,
    BoundingBox,
    Class,
    Classes,
    Enum,
    Float,
    FloatRange,
    Image,
    Integer,
    IntegerRange,
    List,
    Point,
    String,
    Timestamp,
    Tracklet,
)

from rvai.cells.yolov5.algorithm.models.yolo import Model as YoloModel
from rvai.cells.yolov5.algorithm.utils.general import (
    check_anchors,
    check_img_size,
    compute_loss,
    non_max_suppression
)
from rvai.cells.yolov5.algorithm.utils.torch_utils import (
    ModelEMA,
    init_seeds,
    intersect_dicts,
    select_device,
    set_half_prec
)
from rvai.cells.yolov5.yolov5_helpers import (
    create_custom_dataloader,
    evaluate_model,
    get_boxes,
    get_model_yaml,
    load_hyp,
    pre_process,
    send_training_update,
)
from rvai.cells.deep_sort.utils import extract_image_patch, track_to_tracklet
from rvai.cells.deep_sort.structures import Detection
from rvai.cells.deep_sort.deep_sort import TrackerState
from rvai.cells.deep_sort.track import Track, TrackIDGenerator
from rvai.cells.deep_sort.tracker import Tracker

from tracking_dataset import TrackerDataset

from utils import visualize_tracker, mota

class YoloTracker:
    def __init__(self,
                 tar_path: Path,
                 yolo_params: dict,
                 deepsort_params: dict):
        self.tar_path = tar_path
        self.yolo_params = yolo_params
        self.deepsort_params = deepsort_params

        # Initialize a tracker state for DeepSORT
        self.state = TrackerState(
            tracks=[],
            id_gen=TrackIDGenerator()
        )

        self.yolo_model = None
        self.deepsort_model = None

    def load_models(self, tar_path: Path):
        # Load Yolo Model

        yaml_path = get_model_yaml(Enum[String]("Yolov5s", "Yolov5m", "Yolov5l", "Yolov5x", selected=self.yolo_params['variant']))
        yolol, yolom, yolos, deepsort = unpack_model('yolo', Path(tar_path), n_models=4)
        model_paths = {
            "Yolov5s": yolos,
            "Yolov5m": yolom,
            "Yolov5l": yolol
        }
        model_path = model_paths[self.yolo_params['variant']]

        yolo_model = YoloModel(yaml_path, nc=self.yolo_params['n_classes'])
        device = select_device("")
        yolo_model = yolo_model.to(device)
        state_dict = torch.load(
            model_path, map_location=lambda storage, loc: storage
        )
        yolo_model.load_state_dict(state_dict, strict=True)
        yolo_model.float()
        yolo_model = yolo_model.fuse()
        yolo_model = yolo_model.eval()
        if device.type == "cuda":
            set_half_prec(yolo_model, half_prec=self.yolo_params['half_prec'])

        self.yolo_model = yolo_model

        # Load DeepSORT model

        default_weights = DeepSORTCell._get_default_weights()
        deepsort_model = DeepSORTModel(
            model_path = default_weights,
            input_name='images',
            output_name='features'
        )

        self.deepsort_model = deepsort_model

    def _yolo_predict(self, image):

        orig_image = Image(image).as_rgb()

        new_image, ratio_w, ratio_h, pad_w, pad_h = pre_process(
            image=image,
            new_shape=(self.yolo_params['inp_size'], self.yolo_params['inp_size']),
            full_pad=self.yolo_params['full_pad']
        )

        device = select_device("")
        input_tensor = torch.from_numpy(new_image).unsqueeze(0).to(device)

        # When running on cuda, take into account half precision mode
        if device.type == "cuda":
            self.yolo_model = self.yolo_model.cuda()
            set_half_prec(self.yolo_model, half_prec=bool(self.yolo_params['half_prec']))
            if self.yolo_params['half_prec']:
                input_tensor = input_tensor.half()

        # Run prediction
        with torch.no_grad():
            pred, _ = self.yolo_model(input_tensor)

        # Apply NMS
        pred = non_max_suppression(
            pred, self.yolo_params['conf_thresh'], self.yolo_params['iou_thresh']
        )

        class_to_idx, idx_to_class = self.yolo_params['classes'].class_index_mapping()

        # Translate predictions to RVAI bounding_boxes
        boxes = get_boxes(
            pred=pred,
            ratio_width=ratio_w,
            ratio_height=ratio_h,
            pad_width=pad_w,
            pad_height=pad_h,
            target_shape=orig_image.shape[:2],
            index_to_class=idx_to_class,
        )

        return boxes

    def _deepsort_predict(self, image, boxes):

        if self.deepsort_params['use_features']:
            patch_shape = self.deepsort_model.get_input_shape()
            # Get patches
            patches = []
            for box in boxes:
                patches.append(extract_image_patch(image, box, patch_shape))
            # Get the features from the model
            if len(patches) > 0:
                features = self.deepsort_model.predict_batch(patches)
            else:
                features = []
        else:
            features = [None for b in boxes]
        # Create Detection objects
        detections = []
        for box, feat in zip(boxes, features):
            detections.append(Detection(detected_box=box, embedding=feat))
        # Run tracking
        return perform_tracking(
            parameters=self.deepsort_params,
            state=self.state,
            detections=detections,
        )

    def predict(self, image: Image):
        bboxes = self._yolo_predict(image)
        tracklet_list = self._deepsort_predict(image, bboxes)

        return tracklet_list, bboxes


def perform_tracking(parameters: dict,
                     state: TrackerState,
                     detections: list):
    """Basically stolen from rvai, with some adaptations to work around the need for an InferenceContext"""

    # Get the current tracks from the state
    tracks: Sequence[Track] = state.tracks
    id_gen: TrackIDGenerator = state.id_gen

    timestamp = Timestamp()
    # Run a tracker update step
    Tracker.predict(tracks=tracks, timestamp=timestamp)

    # Run an update step
    res_tracks, new_tracks, deleted_tracks = Tracker.update(
        tracks=tracks,
        detections=detections,
        timestamp=timestamp,
        id_gen=id_gen,
        iou_threshold=parameters['iou_threshold'],
        embedding_threshold=parameters['embedding_threshold'],
        use_medoids=parameters['use_medoids'],
        num_medoids=parameters['num_medoids'],
        min_init=parameters['min_init'],
        max_missing=parameters['max_missing'],
        max_tracklet_size=parameters['max_tracklet_size'],
        metric_function=parameters['metric_function'],
    )

    # Store the new tracks in the state
    state.tracks = res_tracks

    tracklets = []
    # Generate tracklet output from the track objects
    max_output_misses = parameters['max_output_misses']
    for track in res_tracks:
        if track.is_confirmed() and track.misses <= max_output_misses:
            tracklets.append(
                track_to_tracklet(
                    track, parameters['tracklet_granularity']
                )
            )

    return List[Tracklet](tracklets)

def visualize_predicted_boxes(image: Image, boxes: List[BoundingBox]):
    image = np.array(image)

    for box in boxes:
        x1, y1 = int(box.p1.x), int(box.p1.y)
        x2, y2 = int(box.p2.x), int(box.p2.y)

        image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), thickness=5)

    plt.figure()
    plt.imshow(image)

if __name__ == '__main__':
    # CONFIGURE HERE
    # --------------------------------------------------------------------------------------

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    tar_path = "/home/laurens/Downloads/Yolotracking(3).tgz"
    YOLO_PARAMS = {
        'variant': "Yolov5s",
        'n_classes': 1,
        'inp_size': 1088,
        'full_pad': False,
        'half_prec': False,
        'conf_thresh': 0.4,
        'iou_thresh': 0.5,
        'classes': Classes([
            Class(class_uuid='Person', name='Person')
        ])
    }

    DEEPSORT_PARAMS = {
        'iou_threshold': 0.75,
        'embedding_threshold': 2.0,
        'use_medoids': False,
        'num_medoids': 1,
        'min_init': 2,
        'max_missing': 6,
        'max_tracklet_size': 60.0,
        'metric_function': "cosine",
        'max_output_misses': 0,
        'use_features': True,
        'tracklet_granularity': 0.5
    }

    # END CONFIGURATION
    # --------------------------------------------------------------------------------------

    # Make a custom tracking pipeline
    tracker = YoloTracker(tar_path,
                          yolo_params=YOLO_PARAMS,
                          deepsort_params=DEEPSORT_PARAMS)

    tracker.load_models(tar_path=tar_path)

    # Make a custom dataset

    data = TrackerDataset(
        img_folder_path=Path('/home/laurens/Documents/MOT20/train/MOT20-01/img1/'),
        annotations_path=Path('/home/laurens/Documents/MOT20/train/MOT20-01/gt/gt.txt'),
        classes_of_interest=[1, 2, 7],
        relabel={1: 14, 2: 14, 7: 14}
    )

    mota_score, true_positives, false_positives, false_negatives, id_switches, switch_dict = mota(data, tracker)

    print(f"MOTA\tTP\tFP\tFN\tID\n{mota_score}\t{true_positives}\t{false_positives}\t{false_negatives}\t{id_switches}")

    pos = true_positives+false_negatives
    print(f"tpr: {true_positives/pos}\tfpr: {false_positives/pos}\tfnr: {false_negatives/pos}\tids: {id_switches/pos}")
    visualize_tracker(tracker, data)

    # for i in range(100):
    #     img, boxes = data[i]
    #     pred_boxes, tracklet_list = tracker.predict(img)
    #     print(len(tracker.state.tracks))

