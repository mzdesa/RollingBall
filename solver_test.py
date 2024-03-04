#Solve optimization problem using Casadi
import numpy as np
import casadi as ca
from scipy.linalg import expm

#set up boundary conditions
R0 = np.eye(3)
RN = np.eye(3)
x0 = np.zeros((2, 1))
xN = np.zeros((2, 1))
N = 20
h = 0.1

#set up optimization problem
opti = ca.Opti()
uk = opti.variable(3, N - 1)
Rk = opti.variable(3, 3*N) #rotation matrices
xk = opti.variable(2, N)
A = ca.MX.zeros(2, 3)
A[0, 1] = 1
A[1, 0] = -1

#set initial guesses to avoid divide by zero
uGuess = ca.DM.ones(3, N-1) 
opti.set_initial(uk, uGuess)
opti.set_initial(xk, ca.DM.ones(2, N))
opti.set_initial(Rk, ca.DM.ones(3, 3*N))


def hat_cas(eta):
    """
    Computes the hat map of a casadi variable eta
    """
    etaHat = ca.MX.zeros(3, 3)
    etaHat[0, 1], etaHat[0, 2] = -eta[2], eta[1]
    etaHat[1, 0], etaHat[1, 2] = eta[2], -eta[0]
    etaHat[2, 0], etaHat[2, 1] = -eta[1], eta[0]
    return etaHat

def expm_ca(v):
    """
    Computes matrix exponential of a skew symmetric matrix
    """
    I = ca.MX.eye(3)
    vNorm = ca.norm_2(v) #2 norm is the same as the vector norm
    vHat = hat_cas(v)
    return I + 1/(vNorm) * ca.sin(vNorm)*vHat + 1/(vNorm**2) * (1 - ca.cos(vNorm)) * vHat @ vHat

#set up optimization constraints on rotation matrices
opti.subject_to(Rk[:, 0:3] == R0) #enforce initial constraint
# opti.subject_to(Rk[:, -1-3:-1] == RN) #enforce terminal constraint
for i in range(N-1):
    #extract matrices
    Ri = Rk[:, 3*i : 3*(i+1)]
    Rip1 = Rk[:, 3*(i+1) : 3*(i+2)]

    #extract input
    ui = uk[:, i]

    #enforce constraint
    # opti.subject_to(Rip1 == expm(hat_cas(ui) * h) @ Ri)
    opti.subject_to(Rip1 == expm_ca(ui * h) @ Ri)

#set up optimization constraints on positions
opti.subject_to(xk[:, 0] ==  x0) #enforce initial constraint
# opti.subject_to(xk[:, -1] == xN) #enforce terminal constraint
for i in range(N-1):
    #extract matrices
    xi = xk[:, i : (i+1)]
    xip1 = xk[:, i+1 : (i+2)]

    #extract input
    ui = uk[:, i]

    #enforce constraint -> use A matrix to modify input
    opti.subject_to(xip1 == xi + h * A @ ui)

cost = ca.norm_2(xk[:, -1] - xN) + ca.norm_fro(Rk[:, -1-3:-1] - RN)

#solve problem
opti.minimize(cost)
option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
opti.solver("ipopt")
sol = opti.solve()

uk = opti.debug.value(uk)
print(uk)

print(sol.xk)