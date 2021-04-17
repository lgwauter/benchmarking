from pathlib import Path
from tracking_dataset import TrackerDataset
import logging
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.utils.data
import datetime

from rvai.base.cell import cell
from rvai.base.context import (
    InferenceContext,
    ModelContext,
    ParameterContext,
    TestContext,
    TrainingContext,
)
from rvai.base.data import (
    Dataset,
    DatasetConfig,
    Expertise,
    Metrics,
    Parameters,
    State,
    Tag,
)
from rvai.base.exc import OOMError
from rvai.base.test import TestSession
from rvai.base.training import ModelPath, TrainingSession, TrainingUpdate
from rvai.base.trtis.cell import TRTISCell
from rvai.base.trtis.client import TRTISClient
from rvai.base.trtis.model import (
    TRTISDataType,
    TRTISInputFormat,
    TRTISInputSpec,
    TRTISModel,
    TRTISModelSpec,
    TRTISOutputSpec,
    TRTISPlatform,
)
from rvai.cells.fairmot.dataset import DatasetWrapper
from rvai.cells.fairmot.datatypes import (
    FairMOTAnnotations,
    FairMOTInputs,
    FairMOTOutputs,
    FairMOTSamples
)
from rvai.cells.fairmot.fairmot import FairMOTParameters, ModelConfig
from rvai.cells.fairmot.fairmot import FairMOTState
from rvai.cells.fairmot.defaults import DEFAULT_HM_DOWN_RATIO, DEFAULT_MAX_LOSS
from rvai.cells.fairmot.model import create_model, load_model, save_model
from rvai.cells.fairmot.structures import Track, TrackIDGenerator
from rvai.cells.fairmot.tracker import Tracker
from rvai.cells.fairmot.training import MOTTrainer
from rvai.cells.fairmot.utils import calculate_map, set_half_prec
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
    String,
    Timestamp,
    Tracklet,
)
from rvai.weights.fairmot import get_weight_resource
from packing import unpack_model
from utils import visualize_tracker, do_bbox_matching, bbox_iou, mota


class FairMOTTracker:
    def __init__(self,
                 tar_path: Path,
                 parameters: FairMOTParameters
                 ):
        self.tar_path = tar_path
        self.parameters = parameters

        # Initialize a tracker state for DeepSORT
        self.state = FairMOTState(
            prev_timestamp = Timestamp(0.0),
            tracks=[],
            id_gen=TrackIDGenerator()
        )

        self.fairmot_model = None
        self.model_config = None

    def load_model(self):
        # Load FairMOT Model

        model_path, _ = unpack_model('Fairmot', Path(self.tar_path), n_models=3)

        # Specify the dimensions of the different heads
        heads = {
            "hm": len(self.parameters.classes),
            "wh": 4
            if not bool(self.parameters.class_specific_size)
            else 4 * len(self.parameters.classes),
            "id": int(self.parameters.reid_dim.value),
            "reg": 2,
        }

        # Create the model
        backbone = self.parameters.backbone.selected.lower()
        model_ = create_model(arch=backbone, heads=heads)

        # Load the model path, if any
        if model_path is not None:
            model_ = load_model(model=model_, model_path=model_path)

        # In training mode, model is transferred to GPU later,
        #  in TRTIS mode we don't need GPU over here.
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model_ = model_.to(device)
        # When on cuda, set half precision mode
        if torch.cuda.is_available():
            set_half_prec(model_, half_prec=bool(self.parameters.half_prec))
        # Set the model to inference mode
        model_.eval()

        self.fairmot_model = model_

        # Create a class to index and reverse mapping, based on sorted classes
        sorted_classes = Classes(
            sorted(self.parameters.classes, key=lambda cl: cl.class_uuid)
        )
        class_to_idx, idx_to_class = sorted_classes.class_index_mapping()
        model_config = ModelConfig(
            class_to_index=class_to_idx, index_to_class=idx_to_class
        )

        self.model_config = model_config

        return model_, model_config

    def predict(self, image):

        timestamp = Timestamp(datetime.datetime.now())
        state = self.state
        tracks = state.tracks
        id_gen = state.id_gen
        prev_timestamp = state.prev_timestamp

        tracks, detections = Tracker.update(
            image=image,
            prev_timestamp=prev_timestamp,
            timestamp=timestamp,
            tracks=tracks,
            model=self.fairmot_model,
            half_prec=bool(self.parameters.half_prec),
            model_input_height=self.parameters.model_input_height.value,
            model_input_width=self.parameters.model_input_width.value,
            conf_threshold=self.parameters.detection_threshold.value,
            motion_cutoff=bool(self.parameters.motion_cutoff),
            feature_lambda=self.parameters.feature_lambda.value,
            position_only=bool(self.parameters.position_only),
            matching_threshold=self.parameters.matching_threshold.value,
            iou_threshold=self.parameters.iou_threshold.value,
            iou_bootstrap_threshold=self.parameters.iou_bootstrap_threshold.value,
            min_init=self.parameters.min_init.value,
            max_time_missing=self.parameters.max_time_missing.value,
            max_tracklet_size=self.parameters.max_tracklet_size.value,
            index_to_class=self.model_config.index_to_class,
            id_gen=id_gen,
        )

        # Update state
        state.tracks = tracks
        state.id_gen = id_gen
        state.prev_timestamp = timestamp

        # Translate Track to tracklet
        tracklets = []
        for track in tracks:
            if track.is_tracked():
                tracklets.append(
                    track.to_rvai_type(
                        granularity_s=self.parameters.tracklet_granularity.value
                    )
                )

        # Only return bounding_boxes
        bounding_boxes: List[BoundingBox] = List[BoundingBox](
            [det.box for det in detections]
        )

        return tracklets, bounding_boxes


if __name__ == "__main__":

    data = TrackerDataset(
        img_folder_path=Path('/home/laurens/Documents/MOT20/train/MOT20-01/img1/'),
        annotations_path=Path('/home/laurens/Documents/MOT20/train/MOT20-01/gt/gt.txt'),
        classes_of_interest=[1, 2, 7],
        relabel={1: 14, 2: 14, 7: 14}
    )

    parameters = FairMOTParameters(
        classes = Classes([
            Class(class_uuid='Person', name='Person')
        ]),
        # MODEL PARAMETERS
        backbone=Enum[String]([], selected="DLA-34"),
        reid_dim=128,
        class_specific_size = False,
        # TRAINING AND RUNTIME PARAMETERS
        model_input_width=1088,
        model_input_height=608,
        # RUNTIME PARAMETERS
        detection_threshold=0.6,
        tracklet_granularity=0.5,
        max_tracklet_size=60.0,
        half_prec=False,
        min_init=2,
        max_time_missing=1.0,
        # Track matching parameters
        motion_cutoff=True,
        feature_lambda=0.98,
        position_only=False,
        matching_threshold=0.4,
        iou_threshold=0.5,
        iou_bootstrap_threshold=0.3,
        )

    tar_path = Path("/home/laurens/Downloads/Fairmottracking(1).tgz")

    tracker = FairMOTTracker(tar_path=tar_path,
                             parameters=parameters)

    model, config = tracker.load_model()

    mota_score, true_positives, false_positives, false_negatives, id_switches, switch_dict = mota(data, tracker)

    print(f"MOTA\tTP\tFP\tFN\tID\n{mota_score}\t{true_positives}\t{false_positives}\t{false_negatives}\t{id_switches}")

    pos = true_positives+false_negatives
    print(f"tpr: {true_positives/pos}\tfpr: {false_positives/pos}\tfnr: {false_negatives/pos}\tids: {id_switches/pos}")

    # visualize_tracker(tracker, data)


    # for i in range(100):
    #     img, bboxes = data[i]
    #     tracker.predict(img)
    #     print(len(tracker.state.tracks))

