import math

import numpy as np

from Side_Functions import *


def UPWINDER(elemNum, Uip, Vip):
    square = squaresQordinations[elemNum]
    ips = XYips[elemNum]

    # finding u and v on ips by bi_lineare
    il = squareNums[elemNum]

    L = 2 * Lmax(square)
    upwind_coef = []
    ds_up_coef = []
    for i in range(0, 4):
        x_ip = ips[i][0]
        y_ip = ips[i][1]
        u = Uip[i]
        v = Vip[i]
        p1 = [x_ip, y_ip]
        if u == 0:
            x2 = x_ip
            y2 = y_ip - np.sign(v) * L
            if v == 0:
                row = [0.25, 0.25, 0.25, 0.25]
                break
        else:
            m = v / u
            LMparams = L / math.sqrt(m ** 2 + 1)
            if np.sign(u) > 0:
                LMparams = -LMparams
            x2 = LMparams + x_ip
            y2 = LMparams * m + y_ip
        p2 = (x2, y2)
        # yesar ro ip yesar 2Lmax oonvartar
        GoP1 = [p1, p2]
        upwind_point_ip = []
        row = np.zeros(4)
        for i in range(0, 4):
            p1 = square[i]
            j = i + 1
            if j == 4: j = 0
            p2 = square[j]
            GoP2 = [p1, p2]
            thePoint = upwind_intersectionFinder(GoP1, GoP2)
            if thePoint:
                upwind_point_ip = thePoint
                row[i] = distance(p2, thePoint, "a") / distance(p1, p2, "h")
                row[j] = distance(p1, thePoint, "L") / distance(p1, p2, "l")
                break
        upwind_coef.append(row)
        if upwind_point_ip == []:
            upwind_point_ip = square[0]
        ds_up_coef.append(distance(upwind_point_ip, p1, [p2, u, v, square]))
    return upwind_coef, ds_up_coef


# dnds dndt of volumes center of sub cv
# dndx dndy and volumes for each ip
def DNDXY(X, Y):
    volumes = np.zeros(4)
    dndx = np.zeros((4, 4))
    dndy = np.zeros((4, 4))

    for ip in range(0, 4):
        temp = dndx_dndy_vol_ip(ip, X, Y)
        dndx[ip] = temp[0]
        dndy[ip] = temp[1]
        volumes[ip] = temp[2]

    return dndx, dndy, volumes


# atention for dsx k and j  + -


def DIFFUS(dsx_el, dsy_el, dndx, dndy, gamma):
    C = np.zeros((4, 4))
    for i in range(0, 4):
        j = i
        k = j - 1
        if j == 0:
            k = 3
        for m in range(0, 4):
            C[i][m] = -gamma * (
                    dndx[i][m] * (dsx_el[j]) + dndy[i][m] * (dsy_el[j]) - dndx[k][m] * dsx_el[k] - dndy[k][m] *
                    dsy_el[k])
    return C


def TRANS(Vol, dt, ro, uold, vold):
    ct = np.zeros((4, 4))
    ddu = np.zeros(4)
    ddv = np.zeros(4)
    for i in range(0, 4):
        ct[i][i] = ro * (Vol[i] / dt)
        # tavajoh be -
        ddu[i] = ro * uold[i] * Vol[i] / dt
        ddv[i] = ro * vold[i] * Vol[i] / dt
    return ct, ddu, ddv


def PRESS(dsx_el, dsy_el):
    dpu = np.zeros((4, 4))
    dpv = np.zeros((4, 4))
    for i in range(0, 4):
        j = i
        k = j - 1
        if j == 0:
            k = 3
        for m in range(0, 4):
            dpu[i][m] = N[j][m] * dsx_el[j] + N[k][m] * (-dsx_el[k])
            dpv[i][m] = N[j][m] * dsy_el[j] + N[k][m] * (- dsy_el[k])
    return dpu, dpv


def Solver(A, B):
    # B=[el*-1 for el in B]
    n = len(points)
    P = np.zeros(n)
    U = np.zeros(n)
    V = np.zeros(n)
    newPUV = np.linalg.solve(A, B)
    for i in range(0, n):
        P[i] = newPUV[i * 3]
        U[i] = newPUV[i * 3 + 1]
        V[i] = newPUV[i * 3 + 2]
    return P, U, V


def DUVPDXY(dndx, dndy, p, u, v):
    dudy = np.zeros(4)
    dvdy = np.zeros(4)
    dudx = np.zeros(4)
    dvdx = np.zeros(4)
    dpdx = np.zeros(4)
    dpdy = np.zeros(4)

    for i in range(0, 4):
        for m in range(0, 4):
            dudx[i] = dudx[i] + dndx[i][m] * u[m]
            dudy[i] = dudy[i] + dndy[i][m] * u[m]
            dvdx[i] = dvdx[i] + dndx[i][m] * v[m]
            dvdy[i] = dvdy[i] + dndy[i][m] * v[m]
            dpdx[i] = dpdx[i] + dndx[i][m] * p[m]
            dpdy[i] = dpdy[i] + dndy[i][m] * p[m]
    return dudy, dvdy, dudx, dvdx, dpdx, dpdy


def UVHAT(dsx, dsy, dndx, dndy, p_elem, u_elem, v_elem, uip, vip, Ds_ups, L_diffusions):
    alpha = np.zeros(4)
    beta = np.zeros(4)
    omega = np.zeros(4)
    mdot = np.zeros(4)

    dudy, dvdy, dudx, dvdx, dpdx, dpdy = DUVPDXY(dndx, dndy, p_elem, u_elem, v_elem, )

    u_hat = np.zeros(4)
    v_hat = np.zeros(4)
    for i in range(0, 4):
        u_hat[i] = uip[i] - (
                    1 / (ro * math.sqrt(uip[i] ** 2 + vip[i] ** 2) / Ds_ups[i] + mio / (L_diffusions[i] ** 2))) * (
                           -ro * (uip[i] * dvdy[i] - vip[i] * dudy[i]) + dpdx[i])
        v_hat[i] = vip[i] - (
                    1 / (ro * math.sqrt(uip[i] ** 2 + vip[i] ** 2) / Ds_ups[i] + mio / (L_diffusions[i] ** 2))) * (
                           -ro * (vip[i] * dudx[i] - uip[i] * dvdx[i]) + dpdy[i])

    for i in range(0, 4):
        alpha[i] = ro * (math.sqrt(uip[i] ** 2 + vip[i] ** 2)) / Ds_ups[i] + gamma / (L_diffusions[i] ** 2)
        beta[i] = uip[i] * dvdy[i] - vip[i] * dudy[i]
        omega[i] = vip[i] * dudx[i] - uip[i] * dvdx[i]
        mdot[i] = ro * u_hat[i] * dsx[i] + ro * v_hat[i] * dsy[i]

    chcu = np.zeros((4, 4))
    chcv = np.zeros((4, 4))
    chp = np.zeros((4, 4))
    dhc = np.zeros(4)

    for i in range(0, 4):
        j = i
        k = j - 1
        if j == 0:
            k = 3
        for m in range(0, 4):
            chcu[i][m] = ro * dsx[i] * N[i][m] + ro * (-dsx[k]) * N[k][m]
            chcv[i][m] = ro * dsy[i] * N[i][m] + ro * (-dsy[k]) * N[k][m]
            chp[i][m] = -(ro * dsx[j] / alpha[j] * dndx[j][m] + ro * dsy[j] / alpha[j] * dndy[j][m]) - (
                    ro * (-dsx[k]) / alpha[k] * dndx[k][m] + ro * (-dsy[k]) / alpha[k] * dndy[k][m])

        dhc[i] = -(ro * dsx[j] * beta[j] / alpha[j] + ro * dsy[j] * omega[j] / alpha[j] + ro * (-dsx[k]) * beta[k] /
                   alpha[k] + ro * (-dsy[k]) * omega[k] / alpha[k])

    return chcu, chcv, chp, dhc, mdot


def UV_COEF(mdot, dsx_el, dsy_el, dndx, dndy, vols, gamma, ro, uip, vip, cUpwinds, dsUP, Ldiff):
    cc = np.zeros((4, 4))
    cpu = np.zeros((4, 4))
    cpv = np.zeros((4, 4))

    for i in range(0, 4):
        j = i
        k = j - 1
        if j == 0:
            k = 3
        peclet_j = ro * math.sqrt(uip[j] ** 2 + vip[j] ** 2) * dsUP[j] / gamma
        peclet_k = ro * math.sqrt(uip[k] ** 2 + vip[k] ** 2) * dsUP[k] / gamma
        Rk = dsUP[k] ** 2 / Ldiff[k] ** 2
        Rj = dsUP[j] ** 2 / Ldiff[j] ** 2
        for m in range(0, 4):
            mdotj = mdot[j]
            mdotk = -mdot[k]
            cc[i][m] = mdotj * (
                    peclet_j / (Rj + peclet_j) * cUpwinds[j][m] + Rj / (Rj + peclet_j) * N[j][m]) + mdotk * (
                               peclet_k / (Rk + peclet_k) * cUpwinds[k][m] + Rk / (Rk + peclet_k) * N[k][m])
            cpu[i][m] = mdotj * (dsUP[j] ** 2 / gamma) / (Rj + peclet_j) * (-dndx[j][m]) + mdotk * (
                        dsUP[k] ** 2 / gamma) / (
                                Rk + peclet_k) * (-dndx[k][m])
            cpv[i][m] = mdotj * (dsUP[j] ** 2 / gamma) / (Rj + peclet_j) * (-dndy[j][m]) + mdotk * (
                        dsUP[k] ** 2 / gamma) / (
                                Rk + peclet_k) * (-dndy[k][m])
    return cc, cpu, cpv
