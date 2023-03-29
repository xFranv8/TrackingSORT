import cv2
from Detector import Detector
from norfair import Detection, Tracker, Video, draw_tracked_objects
import norfair
import numpy as np
import pickle
import torch
from typing import List


def center(points):
    return [np.mean(np.array(points), axis=0)]


PATH = "images/"


def main():



if __name__ == '__main__':
    main()
