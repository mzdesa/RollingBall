import numpy as np
import CalSim as cs
import casadi as ca
from scipy.linalg import logm

class FFController(cs.Controller):
    """
    Simple FF controller for debugging
    """
    def __init__(self, observer, lyapunovBarrierList=None, trajectory=None, depthCam=None):
        """
        Init function for a min norm to full state linearization controller
        Inputs:
            observer (StateObserver): standard state observer
        """
        #first, call super init function on controller class
        super().__init__(observer, lyapunovBarrierList=None, trajectory=None, depthCam=None)
    
    def eval_input(self, t):
        self._u = np.array([[1, 0, 0]]).T
        return self._u
    

class MPCBall(cs.Controller):
    """
    MPC controller over NH dynamics
    """
    def __init__(self, observer, lyapunovBarrierList=None, trajectory=None, depthCam=None):
        """
        Init function for a min norm to full state linearization controller
        Inputs:
            observer (StateObserver): standard state observer
        """
        #first, call super init function on controller class
        super().__init__(observer, lyapunovBarrierList=None, trajectory=None, depthCam=None)

        #set desired Rotation variable
        self.Rd = cs.calc_Rx(np.pi/4)
        self.observer.dynamics.Rd = self.Rd

        #set desired position
        self.xD = np.ones((2, 1))

        #set horizon/discretization params
        self.N = 20
        self.h = 1/50

        #Set weight matrices
        self.QNr = 10 * np.eye(2)
        self.QNxi = 10 * np.eye(3)
        self.Qr = 10 * np.eye(2)
        self.Qxi = 10 * np.eye(3)
        self.R = 0.001 * np.eye(3)

        #set casadi A matrix
        A = ca.MX.zeros(2, 3)
        A[0, 1] = 1
        A[1, 0] = -1
        self.A = A

    def eval_input(self, t):
        #get current state
        x = self.observer.get_state()
        r = x[0: 2, 0].reshape((2, 1))
        R0 = x[2:, 0].reshape((3, 3)).T

        #set up boundary conditions
        x0 = r
        xi0 = np.zeros((3, 1))

        #compute the log of RN to get the boundary conditions for omega
        xD = self.xD
        xiD = cs.vee_3d(logm(self.Rd @ R0.T))

        #set up optimization problem
        opti = ca.Opti()
        uk = opti.variable(3, self.N - 1)
        xik = opti.variable(3, self.N)
        xk = opti.variable(2, self.N)

        #set up boundary conditions
        opti.subject_to(xk[:, 0] ==  x0)
        opti.subject_to(xik[:, 0] == xi0)

        #set up x dynamics in terms of u
        for j in range(self.N-1):
            #extract vectors
            xj = xk[:, j]
            xjp1 = xk[:, j+1]
            xij = xik[:, j]
            xijp1 = xik[:, j+1]

            #extract input
            uj = uk[:, j]

            #enforce constraint -> use A matrix to modify input
            opti.subject_to(xjp1 == xj + self.h * self.A @ uj)
            opti.subject_to(xijp1 == xij + uj*self.h)

        #minimize a squared norm over the horizon
        termCost = (xik[:, -1] - xiD).T @ self.QNxi @ (xik[:, -1] - xiD) + (xk[:, -1] - xD).T @ self.QNr @ (xk[:, -1] - xD)
        cost = termCost
        for i in range(self.N-1):
            cost = cost + (xik[:, i] - xiD).T @ self.Qxi @ (xik[:, i] - xiD) + (xk[:, i] - xD).T @ self.Qr @ (xk[:, i] - xD) + uk[:, i].T @ self.R @ uk[:, i]

        #solve problem
        opti.minimize(cost)
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", option)
        sol = opti.solve()
        self._u = sol.value(uk)[:, 0].reshape((3, 1))
        print(sol.value(cost))
        return self._u
