import numpy as np
import CalSim as cs
import casadi as ca

#determine the theta angles
theta1 = np.pi/5
theta2 = np.pi/3
theta3 = np.pi/4

#set up the gamma function
def calc_R(gamma1, gamma2, gamma3, theta1, theta2):
    """
    Calculate the full rotation matrix given gamma_i, theta_i
    """
    return (cs.calc_Rz(gamma1) @ cs.calc_Ry(theta1) 
            @ cs.calc_Rz(gamma2) @ cs.calc_Rx(theta2) 
            @ cs.calc_Rz(gamma3))

Rd = cs.calc_Rx(np.pi/2) @ cs.calc_Ry(np.pi/2) @ cs.calc_Rz(-np.pi)

#solve for the gamma variables
opti = ca.Opti()
gamma1 = opti.variable()
gamma2 = opti.variable()
gamma3 = opti.variable()

#set up cost function
cost = ca.trace(np.eye(3) - Rd.T @ calc_R(gamma1, gamma2, gamma3, theta1, theta2))
# opti.subject_to(0 <= gamma1)
# opti.subject_to(0 <= gamma2)
# opti.subject_to(0 <= gamma3)
# opti.subject_to(2*np.pi >= gamma1)
# opti.subject_to(2*np.pi >= gamma2)
# opti.subject_to(2*np.pi >= gamma3)

#perform minimization
opti.minimize(cost)
option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
opti.solver("ipopt", option)
sol = opti.solve()

print(sol.value(gamma1))
print(sol.value(gamma2))
print(sol.value(gamma3))
print(sol.value(cost))

print(calc_R(sol.value(gamma1), sol.value(gamma2), sol.value(gamma3), theta1, theta2))
print(Rd)