"""Run this script to perform throughput benchmarking"""
import os
from collections import defaultdict

import cv2
import numpy as np
from rvai.base.pipeline import PipelineFactory
from rvai.base.data import (
    Inputs,
    Outputs,
    Parameters,
    State
)
from rvai.base.runtime import Inference, init

import timeit

from rvai.types import(
    Class,
    Classes,
    Enum,
    Float,
    Image,
    Integer,
    BoundingBox,
    String,
    FloatRange,
    IntegerRange,
    Boolean
)

from rvai.pipelines.yolov5_tracking import Yolov5TrackingPipeline
from rvai.pipelines.fairmot_tracking import FairMOTTrackingPipeline

from rvai.cells.yolov5.yolov5_config import Yolov5Parameters
from rvai.cells.deep_sort.deep_sort import TrackerParameters
from rvai.cells.fairmot.fairmot import FairMOTParameters

from packing import unpack_model
from pathlib import Path

from rvai.extensions.drivers.source_sink_driver import SourceSinkDriver
from rvai.extensions.sources.dummy_source import DummySource
from rvai.extensions.sinks.dummy_sink import DummySink

from rvai.extensions.sources.videostream_source import VideostreamSource
from rvai.extensions.sinks.video_sink import VideoSink

from rvai.extensions.sources.pyav_videoreader import PyAVVideoReader
from rvai.extensions.sinks.mjpeg_sink import MJPEGSink

from rvai.base.drivers import Source, Sink

import time

from tracking_dataset import TrackerDataset
from rvai.base.resources import CellResources

#import tensorflow as tf

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

def benchmark_task(runtime, task, benchmark_images, replicas=None, n_iter=50):
    # Start an inference process for the task
    proc = runtime.start_inference(task)

    print("Starting new benchmark task ...")

    warmup = 10

    # Scale if needed
    if replicas is not None:
        proc.set_replicas(replicas)
    # Do a couple of predictions or warmup
    for i in range(warmup):
        out = proc.predict({"image": benchmark_images[i]}).result()

    # Do throughput measurements
    start = time.time()

    # Do prediction requests

    for i in range(warmup, warmup+n_iter):
        fut = proc.predict({"image": benchmark_images[i]})
    # Get the last result
    fut.result()
    stop = time.time()

    # Estimate throughput
    print(f"Did {n_iter} iterations in {stop-start} seconds time")
    throughput = n_iter / (stop - start)

    # Measure the latency
    latency = timeit.timeit(proc.predict({"image": benchmark_images[0]}).result, number=n_iter)

    # Stop the process
    proc.stop()

    print(throughput, latency)

    return throughput, latency


def benchmark(runtime, pipeline, models, image_sizes, n_iter, dataset, parameters, tracker_type='yolo', replicas=None):
    results = defaultdict(list)
    for img_name, image_size in image_sizes.items():
        # Prepare data for testing
        images = [Image(cv2.resize(np.array(dataset[idx][0], dtype='uint8'), image_size)) for idx in range(len(dataset))]
        results[img_name] = []

        if tracker_type == 'yolo':
            parameters['detector'].img_size = IntegerRange(image_size[0])  # Take the largest size
        else:
            parameters['fairmot'].model_input_width = IntegerRange(image_size[0])
            parameters['fairmot'].model_input_height = IntegerRange(image_size[1])
        
        resources = {
            'detector': CellResources(gpus=0.5),
            'tracker': CellResources(gpus=0.5)
        } if tracker_type == 'yolo' else \
            {'fairmot': CellResources(gpus=1.0)}

        inference = Inference(
            pipeline=pipeline,
            parameters=parameters,
            models=models,
            resources=resources
        )

        # Perform benchmarking
        res = benchmark_task(runtime, inference, images, replicas=replicas, n_iter=n_iter)
        results[img_name].append(res)
    return results


if __name__ == '__main__':

    # Configurations here
    # --------------------------------------------------

    TRACKER_TYPE = "yolo"

    runtime_type = "ray"  # ray, debug

    image_sizes = {  # The image sizes we wish to test
        "480p": (852, 480),
        "720p": (1280, 720),
        "1080": (1920, 1080)
    }

    YOLO_TAR_PATH = Path("/home/laurens/Yolotracking(3).tgz")
    FAIRMOT_TAR_PATH = Path("/home/laurens/Fairmottracking(1).tgz")

    YOLO_PARAMETERS = Yolov5Parameters(
        classes=Classes([Class(class_uuid='Person', name='Person')]),
        yolov5_type=Enum[String](["Yolov5s",
                                  "Yolov5m",
                                  "Yolov5l",
                                  "Yolov5x"], selected="Yolov5l"),
        conf_thresh=FloatRange(value=0.4),
        iou_thresh=FloatRange(value=0.5),
        half_prec=Boolean(False),
        full_pad=Boolean(False),
        img_size=IntegerRange(1088)
    )

    DEEPSORT_PARAMETERS = TrackerParameters(
        iou_threshold=FloatRange(0.75),
        embedding_threshold=FloatRange(2.0),
        use_medoids=Boolean(False),
        num_medoids=IntegerRange(1),
        min_init=IntegerRange(2),
        max_missing=IntegerRange(6),
        max_tracklet_size=FloatRange(60.0),
        metric_function=Enum[String](["euclidian",
                                      "cosine"], selected="cosine"),
        max_output_misses=IntegerRange(0),
        use_features=Boolean(True),
        tracklet_granularity=FloatRange(0.5)
    )

    FAIRMOT_PARAMETERS = FairMOTParameters(
        classes=Classes([
            Class(class_uuid='Person', name='Person')
        ]),
        # MODEL PARAMETERS
        backbone=Enum[String]([String("DLA-34"),
                               String("HRNet-18"),
                               String("HRNet-32")], selected=String("DLA-34")),
        reid_dim=IntegerRange(128),
        class_specific_size=Boolean(False),
        # TRAINING AND RUNTIME PARAMETERS
        model_input_width=IntegerRange(1088),
        model_input_height=IntegerRange(608),
        # RUNTIME PARAMETERS
        detection_threshold=FloatRange(0.6),
        tracklet_granularity=FloatRange(0.5),
        max_tracklet_size=FloatRange(60.0),
        half_prec=Boolean(False),
        min_init=IntegerRange(2),
        max_time_missing=FloatRange(1.0),
        # Track matching parameters
        motion_cutoff=Boolean(True),
        feature_lambda=FloatRange(0.98),
        position_only=Boolean(False),
        matching_threshold=FloatRange(0.4),
        iou_threshold=FloatRange(0.5),
        iou_bootstrap_threshold=FloatRange(0.3),
        )

    # Pipeline creation
    # --------------------------------------------------

    pipeline = Yolov5TrackingPipeline() if TRACKER_TYPE == "yolo" else FairMOTTrackingPipeline()
    tar_path = YOLO_TAR_PATH if TRACKER_TYPE == "yolo" else FAIRMOT_TAR_PATH

    if TRACKER_TYPE == "yolo":
        # Expects the .tar file exported from rvai contains 4 trained models: a small, med and large yolo, and seepSORT
        yolov5l, yolov5m, yolov5s, deepsort = unpack_model(TRACKER_TYPE, tar_path, n_models=4)

        yolo_path = yolov5s if YOLO_PARAMETERS.yolov5_type=="Yolov5s" \
            else yolov5m if YOLO_PARAMETERS.yolov5_type=="Yolov5m" \
            else yolov5l

        models = {"detector": String(os.path.abspath(yolo_path)), "tracker": String(os.path.abspath(deepsort))}
        pipeline_parameters = {"detector": YOLO_PARAMETERS, "tracker": DEEPSORT_PARAMETERS}

    else:
        # Expects the .tar file exported from rvai contains 1 trained model: FairMOT
        fairmot = unpack_model(TRACKER_TYPE, tar_path, n_models=1)
        models = {"fairmot": String(os.path.abspath(*fairmot))}
        pipeline_parameters = {"fairmot": FAIRMOT_PARAMETERS}

    # Initialize a runtime
    # --------------------------------------------------

    debug_rt = init('debug')
    #ray_rt = init('ray')

    # Make a dataset
    # --------------------------------------------------

    data = TrackerDataset(
        img_folder_path=Path('/home/laurens/MOT/'),
        annotations_path=Path('/home/laurens/gt.txt'),
        classes_of_interest=[1, 2, 7],
        relabel={1: 14, 2: 14, 7: 14}
    )
    
    import tensorflow as tf
    import time
    print(f"GPU available: {tf.test.is_gpu_available()}")
    time.sleep(10)

    # Do the benchmarking
    # ---------------------------------------------------

    debug_results = benchmark(debug_rt, pipeline, models, image_sizes,
                              n_iter=50, dataset=data, parameters=pipeline_parameters,
                              tracker_type=TRACKER_TYPE)

    #ray_results = benchmark(ray_rt, pipeline, models, image_sizes,
    #                          n_iter=50, dataset=data, parameters=pipeline_parameters,
    #                          tracker_type=TRACKER_TYPE)
    
    print(debug_results)
