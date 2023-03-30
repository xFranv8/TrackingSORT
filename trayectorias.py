import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from Straight import Straight
from Tracker import Tracker


def main():
    p1_1 = [131, 581]
    p1_2 = [457, 219]

    p2_1 = [156, 529]
    p2_2 = [632, 714]

    p3_1 = [507, 203]
    p3_2 = [982, 401]

    p4_1 = [738, 719]
    p4_2 = [957, 398]

    st1 = Straight(p1_1, p1_2)
    st2 = Straight(p2_1, p2_2)
    st3 = Straight(p3_1, p3_2)
    st4 = Straight(p4_1, p4_2)

    list_st = [st1, st2, st3, st4]

    myTracker = Tracker(list_st)

    """
    im = cv.imread('./stmarc_frames/00000001.jpg')
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    
    plt.plot(700, 700, marker= '.', color= 'yellow', alpha = 0.2)
    plt.imshow(im)
    plt.show()
    """

    # create video
    trck: dict = myTracker.track("inputs/video_stm.mp4", True, True)

    im1 = cv.imread("inputs/trayectorias.png")
    im1 = cv.cvtColor(im1, cv.COLOR_BGR2RGB)

    for k, v in trck.items():
        id_ = k
        lista_prueba = trck[id_]
        if len(lista_prueba) > 2:
            x_ini = lista_prueba[0][0][0]
            y_ini = lista_prueba[0][0][1]

            x_f = lista_prueba[-1][0][0]
            y_f = lista_prueba[-1][0][1]

            distance = (x_f - x_ini, y_f - y_ini)
            if np.linalg.norm(distance) > 100:
                for j in range(len(lista_prueba)):
                    x = lista_prueba[j][0][0]
                    y = lista_prueba[j][0][1]
                    plt.plot(x, y, marker='.', color="yellow", alpha=0.15)

    plt.imshow(im1)
    plt.savefig("outputs/trayectorias.png")
    plt.show()

    # results
    mat_res = myTracker.get_tracking_info()

    # print
    myTracker.print_matrix(mat_res)


if __name__ == '__main__':
    main()
