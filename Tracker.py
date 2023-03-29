import cv2
from Detector import Detector
import norfair
import numpy as np
from Straight import Straight
import torch
from typing import List


class Tracker:
    def __init__(self, straights: list[Straight]):
        self.__straights: list[Straight] = straights
        self.__tracking: dict = {}

        for i in range(200):
            self.__tracking[i] = []

        self.__detector = Detector("yolov5s", "cuda:0")
        self.__tracker = norfair.Tracker(distance_function="euclidean", distance_threshold=100)

    def track(self, video_path: str, show: bool = False, save: bool = True) -> None:
        video = norfair.Video(input_path=video_path)

        for frame in video:
            yolo_detections = self.__detector(
                frame,
                conf_threshold=0.25,
                iou_threshold=0.45,
                image_size=800,
                classes=[2, 3, 5, 7]
                # Filtrar por clases, solo queremos detectar vehiculos
            )

            detections = self.__yolo_detections_to_norfair_detections(
                yolo_detections, track_points="centroid"
            )

            tracked_objects = self.__tracker.update(detections=detections)
            norfair.draw_points(frame, detections)
            norfair.draw_tracked_objects(frame, tracked_objects)
            if len(tracked_objects) > 0:
                for tracked_object in tracked_objects:
                    centroid: np.ndarray = tracked_object.estimate
                    id: int = tracked_object.global_id

                    self.__tracking[id].append(centroid)
            if save:
                video.write(frame)
            if show:
                cv2.imshow("", frame)
                if cv2.waitKey(1) == ord('q'):
                    break

    def get_tracking_info(self) -> np.ndarray:
        recta1: Straight = self.__straights[0]
        recta2: Straight = self.__straights[1]
        recta3: Straight = self.__straights[2]
        recta4: Straight = self.__straights[3]

        res_matrix: np.ndarray = np.zeros((4, 4), np.uint8)

        for k, v in self.__tracking.items():
            id_ = k
            lista_prueba = self.__tracking[id_]

            if len(lista_prueba) > 2:
                x_ini = lista_prueba[0][0][0]
                y_ini = lista_prueba[0][0][1]

                x_f = lista_prueba[-1][0][0]
                y_f = lista_prueba[-1][0][1]

                distance = (x_f - x_ini, y_f - y_ini)
                if np.linalg.norm(distance) > 100:
                    if recta1.is_above((x_ini, y_ini)):  # estoy en la zona 1 (por encima de la recta 1)
                        if not recta2.is_above((x_f, y_f)):
                            #tr_12 += 1
                            res_matrix[0][1] += 1
                        elif recta3.is_above((x_f, y_f)):
                            #tr_13 += 1
                            res_matrix[0][2] += 1
                        elif not recta4.is_above((x_f, y_f)):
                            #tr_14 += 1
                            res_matrix[0][3] += 1
                        else:
                            #tr_11 += 1
                            res_matrix[0][0] += 1

                    elif not recta2.is_above((x_ini, y_ini)):  # es decir, por debajo de la recta 2
                        if recta1.is_above((x_f, y_f)):
                            #tr_21 += 1
                            res_matrix[1][0] += 1
                        elif recta3.is_above((x_f, y_f)):
                            #tr_23 += 1
                            res_matrix[1][2] += 1
                        elif not recta4.is_above((x_f, y_f)):
                            #tr_24 += 1
                            res_matrix[1][3] += 1

                    elif recta3.is_above((x_ini, y_ini)):  # es que el coche empieza en la zona 3
                        if recta1.is_above((x_f, y_f)):
                            #tr_31 += 1
                            res_matrix[2][0] += 1
                        elif not recta2.is_above((x_f, y_f)):
                            #tr_32 += 1
                            res_matrix[2][1] += 1
                        elif not recta4.is_above((x_f, y_f)):
                            #tr_34 += 1
                            res_matrix[2][3] += 1
                            #print(id_)

                    elif not recta4.is_above((x_ini, y_ini)):
                        if recta1.is_above((x_f, y_f)):
                            #tr_41 += 1
                            res_matrix[3][0] += 1
                        elif not recta2.is_above((x_f, y_f)):
                            #tr_42 += 1
                            res_matrix[3][1] += 1
                        elif recta3.is_above((x_f, y_f)):
                            #tr_43 += 1
                            res_matrix[3][2] += 1
        return res_matrix

    def print_matrix(self, matrix: np.ndarray) -> None:
        print('----')
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[1]):
                print(f"tr_{i+1}{j+1} = {matrix[i][j]},", end=' ')
            print()

    def __yolo_detections_to_norfair_detections(self, yolo_detections: torch.tensor, track_points: str = "centroid") \
            -> List[norfair.Detection]:
        norfair_detections: List[norfair.Detection] = []

        if track_points == "centroid":
            detections_as_xywh = yolo_detections.xywh[0]
            for detection_as_xywh in detections_as_xywh:
                centroid = np.array(
                    [detection_as_xywh[0].item(), detection_as_xywh[1].item()]
                )
                scores = np.array([detection_as_xywh[4].item()])
                norfair_detections.append(
                    norfair.Detection(
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
                    norfair.Detection(
                        points=bbox, scores=scores, label=int(detection_as_xyxy[-1].item())
                    )
                )

        return norfair_detections
