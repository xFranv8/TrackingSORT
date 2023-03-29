import cv2 as cv
import numpy as np
import pickle
from Straight import Straight
from Tracker import Tracker


# lo que yo quiero: si la primera deteccion de un coche tiene la y por arriba (en python por abajo) de
"""p1_1 = [195, 299, 1]
p1_2 = [723, 267, 1]

p2_1 = [117, 329, 1]
p2_2 = [108, 477, 1]

p3_1 = [612, 281, 1]
p3_2 = [796, 486, 1]

p4_1 = [795, 477, 1]
p4_2 = [216, 487, 1]"""


p1_1 = [195, 299]
p1_2 = [723, 267]

p2_1 = [117, 329]
p2_2 = [108, 477]

p3_1 = [612, 281]
p3_2 = [796, 486]

p4_1 = [795, 477]
p4_2 = [216, 487]


st1 = Straight(p1_1, p1_2)
st2 = Straight(p2_1, p2_2)
st3 = Straight(p3_1, p3_2)
st4 = Straight(p4_1, p4_2)

list_st = [st1, st2, st3, st4]

myTracker = Tracker(list_st)

# create video
myTracker.track("video.mp4", True, True)

# results
mat_res = myTracker.get_tracking_info()

# print
myTracker.print_matrix(mat_res)

print('fin')


