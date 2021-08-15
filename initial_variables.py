import math
import matplotlib.pyplot as plt

import numpy
import numpy as np


ro = 1  # kg/m3

pi = math.pi
# Nx=int(input('Nx'))

Nx=int(input('Welcome!please enter Nx 31 or ..  '))
Ny = Nx
# dt = float(input('dt'))
dt = 1000


ittrexit = 5000
# input('peclet num')
H = 1
W = 1
print('h is', H)

deltaYn = 1 / (Ny - 1)
deltaXn = 1 / (Nx - 1)
uu=1
rey = 1000
rey = float(input('enter they Reynolds 100 or 1000   '))
mio = ro*uu/rey

gamma = mio

Uold = np.zeros(Nx * Ny)
Vold = np.zeros(Nx * Ny)
Ulinearing = np.zeros(Nx * Ny)
Vlinearing = np.zeros(Nx * Ny)
Pold = np.zeros(Nx * Ny)
Plinearing = np.zeros(Nx * Ny)
Un = np.zeros(Nx * Ny)
Vn = np.zeros(Nx * Ny)
P = np.zeros(Nx * Ny)
U = np.zeros(Nx * Ny)
V = np.zeros(Nx * Ny)

middleLine=[]
for i in range(0,Ny):
    middleLine.append(i*Nx+15)

middleLine=[]
for i in range(0,Ny):
    middleLine.append(i*Nx+int((Nx-1)/2))

horizMiddle=[]
for i in range(0,Nx):
    horizMiddle.append(int((Ny-1)*Nx/2+1+i))

rightEdge = []
leftEdge = []

for i in range(1, Ny):
    rightEdge.append(i * Nx)
    leftEdge.append(i * Nx - 1)

leftEdge.append(Ny * Nx - 1)
top = []
bot = []
for i in range(0, Nx):
    top.append(i)
    bot.append((Nx * Ny) - Nx + i)

cornersPadSaatGard = [0, Nx - 1, Ny * Nx - 1, (Ny - 1) * Nx]
edgeszero = []
for i in rightEdge:
    edgeszero.append(i)
for i in leftEdge:
    edgeszero.append(i)
for i in bot:
    edgeszero.append(i)

edges80 = []
edges80 = np.concatenate((leftEdge, rightEdge, bot), axis=0)
# for i in edges80:

for i in top:
    U[i] = 1
