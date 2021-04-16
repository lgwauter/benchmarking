import cv2
import numpy as np
from rvai.types import List, BoundingBox
from tracking_dataset import TrackerDataset
import torch

def write_stream_to_mp4(videostream: np.ndarray, name="yolo.mp4"):
    h, w = videostream.shape[1:3]
    out = cv2.VideoWriter(name, 0x7634706d, 40.0, (w, h))
    for frame in videostream:
        out.write(frame)
    out.release()

def get_box_coords(box: BoundingBox):
    x1, y1 = int(box.p1.x), int(box.p1.y)
    x2, y2 = int(box.p2.x), int(box.p2.y)

    return x1, y1, x2, y2


def draw_bbox(img: np.ndarray, box: BoundingBox, color: tuple):
    x1, y1, x2, y2 = get_box_coords(box)
    img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=4)

    return img

def id_to_unique_color(id: int):
    # Assumed we don't have more than 100 tracks at one time
    np.random.seed(1)
    colors = [tuple(np.random.choice(range(256), size=3)) for _ in range(1000)]
    color = colors[id]

    return tuple([int(val) for val in color])


def visualize_tracker(tracker, dataset):

    processed_frames = []

    for sample, bbox in dataset:
        tracklets, pred_boxes = tracker.predict(sample)
        sample = np.array(sample)
        for track in tracklets:
            track_id = track.id
            color = id_to_unique_color(track_id)
            nboxes = len(track.history)
            print(nboxes)
            boxes = list(track.history.values())
            # Plot track lines in history
            if nboxes >= 2:
                for idx in range(nboxes-1):
                    if idx ==0:
                        continue
                    box1 = boxes[idx]
                    box2 = boxes[idx+1]
                    x1, y1, x2, y2 = get_box_coords(box1)
                    xnext1, ynext1, xnext2, ynext2 = get_box_coords(box2)

                    xm, ym = (x1+x2)//2, (y1+y2)//2
                    xnextm, ynextm = (xnext1+xnext2)//2, (ynext1+ynext2)//2

                    sample = cv2.line(sample, (xm, ym), (xnextm, ynextm), color, thickness=2, lineType=cv2.LINE_4)
                    # Plot the current bbox
                    if idx == nboxes-2:
                        sample = cv2.rectangle(sample, (xnext1, ynext1), (xnext2, ynext2), color, thickness=2)

        cv2.imshow("Vid", cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        processed_frames.append(cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))

    write_stream_to_mp4(np.array(processed_frames, dtype='uint8'))

    cv2.destroyAllWindows()


def mota(dataset: TrackerDataset,
         tracker):

    # Calculates the MOTA for a dataset.
    # MOTA = 1 - sum(misses + false_positives + mismatches) / sum(ground_truths)

    # Loop over the dataset. Make one prediction to warm up the tracker

    img, _ = dataset[0]
    _ = tracker.predict(img)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    id_switches = 0

    id_match_dict = {}

    for idx in range(1, len(dataset)):
        sample, target_boxes = dataset[idx]
        _ = tracker.predict(sample)  # We don't need the outputs, they are in the tracker state
        predicted_boxes = get_bbox_list_with_id_attributes(tracker)

        gt_match_idx, pred_match_idx, gt_ids, pred_ids, tp, fp, fn = do_bbox_matching(predicted_boxes,
                                                                                      target_boxes,
                                                                                      iou_thres=0.5)

        # Update the id match dict
        for gt_id, pred_id in zip(gt_ids, pred_ids):
            if gt_id not in id_match_dict.keys():
                id_match_dict[gt_id] = [pred_id]
            else:
                id_match_dict[gt_id].append(pred_id)

        # Update the number of tp, fp, fn
        true_positives += tp
        false_positives += fp
        false_negatives += fn

    # Calculate the total number of ID switches. An ideal tracker would assign the same ID to the same gt box the
    # entire time. The number of different IDs it generates for one gt box is defined as the number of ID switches.

    for id_list in id_match_dict.values():
        # An ideal tracking has 1 distinct id per gt box.  N distinct ids mean N-1 switches occurred.
        n_distinct = len(set(id_list))
        id_switches += n_distinct-1

    total_positives = true_positives + false_negatives

    mota_score = 1 - (false_negatives + false_positives + id_switches)/(total_positives)

    print(f"Total mota score: {mota_score}")

    return mota_score, true_positives, false_positives, false_negatives, id_switches, id_match_dict


def bbox_iou(pred: BoundingBox,
             gt: BoundingBox):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """

    pred_box = get_box_coords(pred)
    gt_box = get_box_coords(gt)

    # 1.get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area of union
    uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
           (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
           inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni

    return iou


def get_bbox_list_with_id_attributes(tracker) -> List[BoundingBox]:
    """Just as the ground truth data, set the object id the bbox as an attribute."""
    tracks = tracker.state.tracks
    bboxes = []

    for track in tracks:
        try:
            track_id = track.track_uid  # For FairMOT
        except AttributeError:
            track_id = track.uid  # For Yolo / DeepSORT
        bbox = list(track.box_history.values())[-1]
        bbox.set_attributes(object_id=track_id)

        bboxes.append(bbox)

    return List[BoundingBox](bboxes)


def do_bbox_matching(predicted_bboxes: List[BoundingBox],
                     target_bboxes: List[BoundingBox],
                     iou_thres=0.5):
    gt_match_idx = []
    pred_match_idx = []
    gt_ids = []
    pred_ids = []

    for pred_idx, pred_bbox in enumerate(predicted_bboxes):

        # Compute iou with target boxes
        iou = np.array([bbox_iou(pred_bbox, target_bbox) for target_bbox in target_bboxes])

        # Extract index of largest overlap
        gt_idx = np.argmax(iou)
        gt_box = target_bboxes[gt_idx]

        # If overlap exceeds threshold we have a match ladies and gentlemen
        if iou[gt_idx] > iou_thres and gt_idx not in gt_match_idx:
            gt_match_idx.append(gt_idx)
            pred_match_idx.append(pred_idx)
            gt_ids.append(int(gt_box.get_attributes()['object_id']))
            pred_ids.append(int(pred_bbox.get_attributes()['object_id']))

    tp = len(gt_match_idx)
    fp = len(predicted_bboxes) - len(gt_match_idx)
    fn = len(target_bboxes) - len(gt_match_idx)

    print(f"TP: {tp}\tFP: {fp}\t FN: \t{fn}")

    return gt_match_idx, pred_match_idx, gt_ids, pred_ids, tp, fp, fn

def ground_truths(bbox_list: List[BoundingBox]):
    return len(bbox_list)

def map():
    pass

