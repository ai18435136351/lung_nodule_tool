import numpy as np

x1 = 81
x2 = 40

def acc(img2, img1,x,y):
    img3 = np.zeros((x1, x1))
    img4 = np.zeros((x1, x1))
    for i in range(x1):
        for j in range(x1):
            if img1[i+y-x2][j+x-x2]  > 10 and img2[i][j] > 0:
                img3[i][j] = 255
            if img1[i+y-x2][j+x-x2] > 10 or img2[i][j] > 0:
                img4[i][j] = 255
    t = 0
    k = 0
    for i in range(x1):
        for j in range(x1):
            if img3[i][j] == 255:
                t = t + 1
            if img4[i][j] == 255:
                k += 1
    if k == 0:
        return 0
    else:
        return t/(k)

