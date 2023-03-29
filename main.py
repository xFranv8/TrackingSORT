import os

import cv2
import numpy as np


def video_from_frames(path: str, out_path: str) -> None:
    files: list[str] = sorted(os.listdir(path))

    images = []
    for file in files:
        img = cv2.imread(path+file)
        images.append(img)

    height, width = img.shape[:2]

    video = cv2.VideoWriter(out_path + ".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height))

    # For para guardar frames en un video
    for i in range(len(images)):
        video.write(images[i])

    video.release()


PATH = "images/"


def main():
    """."""

    """
    sherbrooke
    
    p1_1 = [63, 294]
    p1_2 = [437, 35]

    p2_1 = [63, 294]
    p2_2 = [619, 572]

    p3_1 = [554, 66]
    p3_2 = [965, 195]

    p4_1 = [1001, 269]
    p4_2 = [730, 571]"""

    """
    rouen
    
    p1_1 = [178, 365]
    p1_2 = [498, 113]

    p2_1 = [158, 377]
    p2_2 = [410, 569]

    p3_1 = [589, 75]
    p3_2 = [902, 226]

    p4_1 = [849, 232]
    p4_2 = [561, 572]"""


    #st_m

    p1_1 = [131, 581]
    p1_2 = [457, 219]

    p2_1 = [156, 529]
    p2_2 = [632, 714]

    p3_1 = [507, 203]
    p3_2 = [982, 401]

    p4_1 = [738, 719]
    p4_2 = [957, 398]

    image = cv2.imread("images_stmarc/00000001.jpg")
    cv2.line(image, p1_1, p1_2, (255, 0, 0), 3)
    cv2.line(image, p2_1, p2_2, (255, 0, 255), 3)
    cv2.line(image, p3_1, p3_2, (255, 255, 0), 3)
    cv2.line(image, p4_1, p4_2, (0, 255, 0), 3)

    cv2.imshow("", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    video_from_frames("./images_stmarc/", "video_stm")


if __name__ == '__main__':
    main()
