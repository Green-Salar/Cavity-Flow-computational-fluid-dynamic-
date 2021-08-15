import math
import numpy as np
import shapely
from shapely.geometry import LineString, Point
import numpy as np
import math
import math
import matplotlib.pyplot as plt
import numpy as np
from Grid_and_ds_generator import *


def distance(p1, p2,p3):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def is_empty(any_structure):
    if any_structure:
        return False
    else:
        return True


def upwind_intersectionFinder(PoL1, PoL2):
    line1 = LineString([PoL1[0], PoL1[1]])
    line2 = LineString([PoL2[0], PoL2[1]])
    D = [line1, line2]
    int_pt = D[0].intersection(D[1])
    if is_empty(int_pt):
        return False
    else:
        point_of_intersection = int_pt.x, int_pt.y
        return point_of_intersection


def Lmax(square):
    x, y = square[0]
    Ymax = y
    Ymin = y
    Xmax = x
    Xmin = x
    for i in square:
        if i[1] > Ymax: Ymax = i[1]
        if i[1] < Ymin: Ymin = i[1]
        if i[0] > Xmax: Xmax = i[0]
        if i[0] < Xmin: Xmin = i[0]
    L = math.sqrt((Xmax - Xmin) ** 2 + (Ymax - Ymin) ** 2)
    return L


def uvpgiver(il):
    Uold_elem = np.zeros(4)
    Vold_elem = np.zeros(4)
    for i in range(0, 4):
        Uold_elem[i] = Uold[il[i]]
        Vold_elem[i] = Vold[il[i]]

    U_elem = np.zeros(4)
    V_elem = np.zeros(4)
    for i in range(0, 4):
        U_elem[i] = Ulinearing[il[i]]
        V_elem[i] = Vlinearing[il[i]]

    Uip_elem = np.zeros(4)+0.01
    Vip_elem = np.zeros(4)+0.01
    for i in range(0, 4):
        for m in range(0, 4):
            Uip_elem[i] = Uip_elem[i] + N[i][m] * U_elem[m]
            Vip_elem[i] = Vip_elem[i] + N[i][m] * V_elem[m]

    P_elem = np.zeros(4)
    for i in range(0, 4):
        P_elem[i] = Plinearing[il[i]]

    return Uold_elem, Vold_elem, U_elem, V_elem, Uip_elem, Vip_elem, P_elem


def dashLinePlotting():
    plt.figure()
    cp = plt.contour(Xs, Ys, T, colors='black', linestyles='dashed')
    plt.clabel(cp, inline=True,
               fontsize=10)
    plt.title('Contour Plot')
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.show()


def contourPlloting():
    fig = plt.figure(figsize=(10, 10))
    left, bottom, width, height = 0.1, 0.1, .8, 0.8
    ax = fig.add_axes([left, bottom, width, height])
    MUV=np.zeros((Nx,Ny))
    for i in range(0,Nx):
        for j in range (0,Nx):
            MUV[i][j]=math.sqrt(U**2+V**2)
    cp = plt.contourf(Xs, Ys, MUV)
    plt.colorbar(cp)

    ax.set_title('Contour Plot')
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    plt.show()


def colorLinePlotitng():
    Xs = np.reshape(xs, (Nx, Ny))
    Ys = np.reshape(ys, (Nx, Ny))
    fig = plt.figure(figsize=(6, 5))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height])
    MUV=np.zeros((Nx,Ny))
    for i in range(0,Nx):
        for j in range (0,Nx):
            MUV[i][j]=math.sqrt(U**2+V**2)
    cp = ax.contour(Xs, Ys, MUV)
    ax.clabel(cp, inline=True,
              fontsize=10)
    ax.set_title('Contour Plot')
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    plt.show()



def vector_field():
    Xs = np.reshape(xs, (Nx, Ny))
    Ys = np.reshape(ys, (Nx, Ny))
    plt.figure()
    cp = plt.quiver(Xs, Ys, U, V, color='g', units='xy')
    plt.title('Vector field')
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.show()

# square midi ba uv va ip - 4x4 tahvil migiri baraye zarayeb (a prime) + 4 taii baraye ds up
# a,coef = square_upwinder([(0.5, 0.5), (-0.8, 0.9), (-0.7, -0.9), (0.6, -0.96)], [(0, 0.5), (0.3, 0), (-0.29, -0.3), (1 /6, -1 / 6)],[(1 / 2, 1 / 2), (1 / 2, -1 / 2), (-1 / 2, -1 / 2), (2 / 3, -2 / 3)])
