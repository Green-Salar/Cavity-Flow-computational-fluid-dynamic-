import shapely
from shapely.geometry import LineString, Point
import numpy as np
import math
import math
import matplotlib.pyplot as plt
import numpy as np

from dndx_dndy_vol_forip import *


# from Upwinder import *


# Given these endpoints

# givven points of lines
def intersectionFinder(PoL1, PoL2):
    line1 = LineString([PoL1[0], PoL1[1]])
    line2 = LineString([PoL2[0], PoL2[1]])
    D = [line1, line2]
    int_pt = D[0].intersection(D[1])
    # print(int_pt)
    point_of_intersection = int_pt.x, int_pt.y
    return point_of_intersection;


XYhorizontalCenter = []
XYverticalCenter = []
XYup = []
XYdown = []
XYleft = []
XYright = []
for x in np.arange(0.5, -0.51, -deltaXn):
    XYup.append((round(x, 4), 0.5))
    XYdown.append((round(x, 4), -0.5))

for y in np.arange(0.5, -0.51, -deltaYn):
    XYright.append((0.5, round(y, 4)))
    XYleft.append((-0.5, round(y, 4)))
# iijad noghat rooye khate bala o paiin

points = []
# i shomarande bala be paiin az 0 shoroo mishe
# for baraye nime balaii
for i in range(0, Ny):
    # noghte avale khat az rast
    points.append(XYright[i])
    # j rast be chap az 0
    for j in range(1, Nx - 1):
        if j > 0 and j < Nx - 1:
            # group of points bala
            GoP1 = [XYright[i], XYleft[i]]
            GoP2 = [XYup[j], XYdown[j]]
            point_of_intersection = intersectionFinder(GoP1, GoP2)
            points.append(point_of_intersection)
        #      print(point_of_intersection)
    # noghte akhare khat
    points.append(XYleft[i])

# khatte vasat va paiin

print('number of points', len(points))

# scaling
for i in range(0, len(points)):
    points[i] = (points[i][0] * W, points[i][1] * H)

squareNums = []
for i in range(0, Ny - 1):
    for j in range(0, Nx - 1):
        firstNum = i * Nx + j
        squareNums.append([firstNum, firstNum + 1, firstNum + Nx + 1, firstNum + Nx])

# print('number of points',len(squareNums))
# print(squareNums[0])

squaresQordinations = []
for i in range(0, len(squareNums)):
    square = []
    for j in squareNums[i]:
        square.append(points[j])
    squaresQordinations.append(square)

markazha = []
L_diffusion_all = []
dsx = []
dsy = []
X_ip_all = []
Y_ip_all = []
XY_ip_all = []
UV_elemans = []
C_upwinds = []
Ds_ups = []
XYips = []

# squaresQordinations =[ [(1, 1), (-1, 1), (-1, -1), (1, -1)]]
# ____ DSX DSY MARKAZ HA X Y U ...
for square in squaresQordinations:
    markaz = []
    Xvasate1 = (square[0][0] + square[1][0]) / 2
    Xvasate3 = (square[2][0] + square[3][0]) / 2
    Yvasate1 = (square[0][1] + square[1][1]) / 2
    Yvasate3 = (square[2][1] + square[3][1]) / 2
    vasate1 = (Xvasate1, Yvasate1)
    vasate3 = (Xvasate3, Yvasate3)
    GoP1 = [vasate1, vasate3]
    # print(GoP1)

    Xvasate2 = (square[1][0] + square[2][0]) / 2
    Xvasate4 = (square[3][0] + square[0][0]) / 2
    Yvasate2 = (square[1][1] + square[2][1]) / 2
    Yvasate4 = (square[3][1] + square[0][1]) / 2
    vasate2 = (Xvasate2, Yvasate2)
    vasate4 = (Xvasate4, Yvasate4)
    GoP2 = [vasate2, vasate4]
    # print(GoP2)
    markaz = intersectionFinder(GoP1, GoP2)
    markazha.append(markaz)
    vasate_azlaa = [vasate1, vasate2, vasate3, vasate4]

    L_diffusion_all.append(L_diffusion(square, vasate_azlaa, markaz))

    dsx1 = (markaz[1] - Yvasate1)
    dsx2 = (markaz[1] - Yvasate2)
    dsx3 = (markaz[1] - Yvasate3)
    dsx4 = (markaz[1] - Yvasate4)
    temp = [dsx1, dsx2, dsx3, dsx4]
    dsx.append(temp)

    dsy1 = - (markaz[0] - Xvasate1)
    dsy2 = - (markaz[0] - Xvasate2)
    dsy3 = - (markaz[0] - Xvasate3)
    dsy4 = - (markaz[0] - Xvasate4)
    temp = [dsy1, dsy2, dsy3, dsy4]
    dsy.append(temp)

    X_ip = np.zeros(4)
    Y_ip = np.zeros(4)

    ips = []
    for ip in range(0, 4):
        for m in range(0, 4):
            X_ip[ip] = X_ip[ip] + N[ip][m] * square[m][0]
            Y_ip[ip] = Y_ip[ip] + N[ip][m] * square[m][1]
        ips.append([X_ip[ip], Y_ip[ip]])

    XYips.append(ips)

#   upwind_coef_4x4, ds_up = square_upwinder(square, ips, UV_in_ip_eleman)
# Upwind_Coefs = yek araye  elemnum*4*4
#  C_upwinds.append(upwind_coef_4x4)
# Ds_ups.append(ds_up)

# X_ip_all.append(X_ip)
# Y_ip_all.append(Y_ip)

# XY_ip.append(X_ip,Y_ip)


# UV_elemans= np.reshape(6*6)
xs = [x[0] for x in points]
ys = [x[1] for x in points]

xd = [x[0] for x in markazha]
yd = [x[1] for x in markazha]

# plt.plot(XYleft,XYup,XYdown,XYright,XYhorizontalCenter,XYverticalCenter)
# plt.scatter(xd, yd, color='red')
# plt.scatter(xs, ys, color='green')
# plt.show()
