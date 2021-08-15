import nt
import shapely
from shapely.geometry import LineString, Point
import numpy as np
import math
import math
import matplotlib.pyplot as plt
import numpy as np
# from Grid_and_ds_generator import *
# from dnds_dndt import *
# grid and ds dx has been generated in th Grid file
# dnds dndt for integration points
from scipy import linalg
# for hame eleman haro bere squares:
from FUNCTIONS import *

err = 300
err0 = 100
ittr = 0
while err0 > .1:
    print("please wait ... time step ...")
    while (err > .1):
        ittr = ittr + 1
        if (ittr == ittrexit): break
        if ittr % 50 == 0: print(ittr)
        elemNum = 0
        A = np.zeros((Nx * Ny * 3, Nx * Ny * 3))
        B = np.zeros(Nx * Ny * 3)
        #go square by square and calculate coefitients and assemble it to the governer matrice
        for square in squaresQordinations:
            X = []
            Y = []
            for XY in square:
                X.append(XY[0])
                Y.append(XY[1])
            dndx, dndy, vols = DNDXY(X, Y)
            il = squareNums[elemNum]
            #initialization our eleman nodal values
            Uold_elem, Vold_elem, U_elem, V_elem, Uip_elem, Vip_elem,P_elem = uvpgiver(il)


            a_prime_coef, ds_up_coef = UPWINDER(elemNum, Uip_elem, Vip_elem)
            ct, ddu, ddv = TRANS(vols, dt, ro, Uold_elem, Vold_elem)
            cd = DIFFUS(dsx[elemNum], dsy[elemNum], dndx, dndy, gamma)
            dpu, dpv = PRESS(dsx[elemNum], dsy[elemNum])

            chcu, chcv, chp, dhc, mdot = UVHAT(dsx[elemNum], dsy[elemNum], dndx, dndy,P_elem, U_elem, V_elem, Uip_elem, Vip_elem,
                                         ds_up_coef, L_diffusion_all[elemNum])

            cc, cpu, cpv = UV_COEF(mdot,dsx[elemNum], dsy[elemNum], dndx, dndy, vols, gamma, ro, Uip_elem, Vip_elem,
                                   a_prime_coef, ds_up_coef, L_diffusion_all[elemNum])

            for j in range(0, 4):
                jj = il[j] * 3

                for k in range(0, 4):
                    kk = il[k] * 3
                    A[jj][kk] = A[jj][kk] + chp[j][k]
                    A[jj][kk + 1] = A[jj ][kk + 1] + chcu[j][k]
                    A[jj][kk + 2] = A[jj ][kk + 2] + chcv[j][k]
                    A[jj + 1][kk + 0] = A[jj + 1][kk + 0] + cpu[j][k] + dpu[j][k]
                    A[jj + 2][kk + 0] = A[jj + 2][kk + 0] + cpv[j][k] + dpv[j][k]
                    A[jj + 1][kk + 1] = A[jj + 1][kk + 1] + cc[j][k] + cd[j][k]
                    A[jj + 2][kk + 2] = A[jj + 2][kk + 2] + cc[j][k] + cd[j][k]

                A[jj + 1][jj + 1] = A[jj + 1][jj + 1] + ct[j][j]
                A[jj + 2][jj + 2] = A[jj + 2][jj + 2] + ct[j][j]

                B[jj + 0] = B[jj + 0] + dhc[j]
                B[jj + 1] = B[jj + 1] + ddu[j]
                B[jj + 2] = B[jj + 2] + ddv[j]
            elemNum = elemNum + 1

        for j in edgeszero:
           jj = j * 3
           for z in range(0, Nx * Ny * 3):
               A[jj + 1][z] = 0
               A[jj + 2][z] = 0
           A[jj+1][jj+1]=1
           A[jj+2][jj+2]=1
           A[jj + 1][jj + 1] = 1
           B[jj + 1] = 0
           A[jj + 2][jj + 2] = 1
           B[jj + 2] = 0


        for j in top:
            jj = j * 3
            for z in range(0, Nx * Ny * 3):
                A[jj + 1][z] = 0
                A[jj + 2][z] = 0
            A[jj + 1][jj + 1] = 1
            B[jj + 1] = 1
            A[jj + 2][jj + 2] = 1
            B[jj + 2] = 0

        for z in range(0,Nx*Ny*3):
            A[Nx*Ny*3-3][z]=0
        A[Nx*Ny*3-3][Nx*Ny*3-3]=1
        B[Nx*Ny*3-3]=1

        Pn, Un, Vn = Solver(A, B)

        err = 0
        for i in range(0, Nx * Ny):
            err = err + abs(Plinearing[i] - Pn[i]) + abs(Ulinearing[i] - Un[i]) + abs(Vlinearing[i] - Vn[i])
            Plinearing[i], Ulinearing[i], Vlinearing[i] = Pn[i], Un[i], Vn[i]
    err0 = 0
    for i in range(0, Nx * Ny):
        err0 = err0 + abs(Plinearing[i] - Pold[i]) + abs(Ulinearing[i] - Uold[i]) + abs(Vlinearing[i] - Vold[i])
        Pold[i], Uold[i], Vold[i] = Plinearing[i], Ulinearing[i], Vlinearing[i]


P, U, V = Pold, Uold, Vold





import scipy.interpolate as interpolate
def streamplot():
    xi = np.linspace(min(xs), max(xs), 100)
    yi = np.linspace(min(xs), max(ys), 100)
    X, Y = np.meshgrid(xi, yi)
    UU = interpolate.griddata((xs, ys), U, (X, Y), method='nearest')
    VV = interpolate.griddata((xs, ys), V, (X, Y), method='nearest')
    plt.figure()
    plt.title('stream-lines and velocity field ')
    plt.xlabel('x ')
    plt.ylabel('y ')
    plt.quiver(xs, ys, U, V, scale_units='xy', angles='xy')
    plt.streamplot(X, Y, UU, VV, color="g", linewidth=1, cmap=plt.cm.autumn)
    plt.show()
streamplot()

def vector_field():
    Xs = np.reshape(xs, (Nx, Ny))
    Ys = np.reshape(ys, (Nx, Ny))
    plt.figure()
    cp = plt.quiver(Xs, Ys, U, V, color='g', units='xy')
    plt.title('Vector field 61')
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.show()

vector_field()

g=2