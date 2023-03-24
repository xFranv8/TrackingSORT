import cv2
from Detector import Detector
from norfair import Detection, Tracker, Video, draw_tracked_objects
import norfair
import numpy as np
import torch
from typing import List


def center(points):
    return [np.mean(np.array(points), axis=0)]


def yolo_detections_to_norfair_detections(
        yolo_detections: torch.tensor, track_points: str = "centroid"  # bbox or centroid
) -> List[Detection]:
    """convert detections_as_xywh to norfair detections"""
    norfair_detections: List[Detection] = []

    if track_points == "centroid":
        detections_as_xywh = yolo_detections.xywh[0]
        for detection_as_xywh in detections_as_xywh:
            centroid = np.array(
                [detection_as_xywh[0].item(), detection_as_xywh[1].item()]
            )
            scores = np.array([detection_as_xywh[4].item()])
            norfair_detections.append(
                Detection(
                    points=centroid,
                    scores=scores,
                    label=int(detection_as_xywh[-1].item()),
                )
            )
    elif track_points == "bbox":
        detections_as_xyxy = yolo_detections.xyxy[0]
        for detection_as_xyxy in detections_as_xyxy:
            bbox = np.array(
                [
                    [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                    [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()],
                ]
            )
            scores = np.array(
                [detection_as_xyxy[4].item(), detection_as_xyxy[4].item()]
            )
            norfair_detections.append(
                Detection(
                    points=bbox, scores=scores, label=int(detection_as_xyxy[-1].item())
                )
            )

    return norfair_detections


PATH = "images/"

detector = Detector("yolov5s",  "cuda:0")

video = Video(input_path="video.mp4")
tracker = Tracker(distance_function="euclidean", distance_threshold=100)

tracking = {}

for frame in video:
    yolo_detections = detector(
                frame,
                conf_threshold=0.25,
                iou_threshold=0.45,
                image_size=800,
                classes=[2, 3, 5, 7]
                # Filtrar por clases, solo queremos detectar vehiculos
            )

    detections = yolo_detections_to_norfair_detections(
        yolo_detections, track_points="centroid"
    )
    """norfair_detections = [Detection(points) for points in detections]
    tracked_objects = tracker.update(detections=norfair_detections)
    draw_tracked_objects(frame, tracked_objects)
    video.write(frame)"""

    tracked_objects = tracker.update(detections=detections)
    norfair.draw_points(frame, detections)
    norfair.draw_tracked_objects(frame, tracked_objects)
    if len(tracked_objects) > 0:
        print(f"IDs: {[tracked_object.global_id for tracked_object in tracked_objects]}")

        # Esta funcion devuelve los centroides estimados en el mismo orden que estan en tracked_objects
        print(f"estimate: {[id.estimate for id in tracked_objects]}")

    cv2.imshow("", frame)
    if cv2.waitKey(0) == ord('q'):
        continue

    # video.write(frame)

    # Lista con ObjectID y Centroide
