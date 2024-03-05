"""
Run discretization solver test for the sphere
"""
import numpy as np
import casadi as ca
import CalSim as cs
from scipy.linalg import logm

#set up boundary conditions
R0 = np.eye(3)
RN = cs.calc_Rx(np.pi/2)
x0 = np.zeros((2, 1))
xD = np.zeros((2, 1))

#compute the log of RN to get the boundary conditions for omega
xiD = cs.vee_3d(logm(RN @ R0.T)) #Not correct! should be 

#set horizon/discretization params
N = 100
h = 1/20

#set up optimization problem
opti = ca.Opti()
uk = opti.variable(3, N - 1)
xik = opti.variable(3, N)
xk = opti.variable(2, N)

A = ca.MX.zeros(2, 3)
A[0, 1] = 1
A[1, 0] = -1

#set up boundary conditions
opti.subject_to(xk[:, 0] ==  x0) #enforce initial constraint
opti.subject_to(xik[:, 0] == np.zeros((3, 1)))
# opti.subject_to(xk[:, -1] ==  xD)
# opti.subject_to(xik[:, -1] == xiD)

#set up x dynamics in terms of u
for j in range(N-1):
    #extract vectors
    xj = xk[:, j]
    xjp1 = xk[:, j+1]
    xij = xik[:, j]
    xijp1 = xik[:, j+1]

    #extract input
    uj = uk[:, j]

    #enforce constraint -> use A matrix to modify input
    opti.subject_to(xjp1 == xj + h * A @ uj)
    opti.subject_to(xijp1 == xij + uj*h)

#minimize a squared norm over the horizon
termCost = (xik[:, -1] - xiD).T @ (xik[:, -1] - xiD) + (xk[:, -1] - xD).T @ (xk[:, -1] - xD)
cost = termCost
for i in range(N-1):
    cost = cost + (xik[:, i] - xiD).T @ (xik[:, i] - xiD) + (xk[:, i] - xD).T @ (xk[:, i] - xD) + uk[:, i].T @ uk[:, i]

#solve problem
# opti.minimize(cost)
opti.solver("ipopt")
sol = opti.solve()
print(sol.value(uk))
print(sol.value(cost))
