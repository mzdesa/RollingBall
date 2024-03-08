import numpy as np
import casadi as ca
import CalSim as cs
from scipy.spatial.transform import Rotation


class MPCBallQuat(cs.Controller):
    """
    MPC controller over quaternion NH dynamics
    Note: A direct optimization over quaternions doesn't seem to work!
    """
    def __init__(self, observer, lyapunovBarrierList=None, trajectory=None, depthCam=None):
        """
        Init function for a min norm to full state linearization controller
        Inputs:
            observer (StateObserver): standard state observer
        """
        #first, call super init function on controller class
        super().__init__(observer, lyapunovBarrierList=None, trajectory=None, depthCam=None)

        #set desired quaternion variable
        Rd = cs.calc_Rz(np.pi/2)
        qdSP = Rotation.from_matrix(Rd).as_quat() #scipy quaternion
        self.qD = np.hstack([qdSP[3], qdSP[0:3]]) #reshape to scalar-first quaternion

        #set desired position
        self.xD = np.ones((2, 1))

        #set horizon/discretization params
        self.N = 60
        self.h = 1/20

        #Set weight matrices
        self.QNr = 5 * np.eye(2)
        self.QNq = 10 * np.eye(4)
        self.Qr = 5 * np.eye(2)
        self.Qq = 10 * np.eye(4)
        self.R = 0.01 * np.eye(3)

    def hat_cas(self, eta):
        """
        Computes the hat map of a casadi variable eta
        """
        etaHat = ca.MX.zeros(3, 3)
        etaHat[0, 1], etaHat[0, 2] = -eta[2], eta[1]
        etaHat[1, 0], etaHat[1, 2] = eta[2], -eta[0]
        etaHat[2, 0], etaHat[2, 1] = -eta[1], eta[0]
        return etaHat

    def eval_A_cas(self, q):
        """
        Evaluates A matrix with casadi MX variables
        """
        #define A matrix as a funcntion of the quaternion
        A = ca.MX.zeros((6, 3))

        #define top elements
        A[0, 0], A[0, 1], A[0, 2] = 0, 1, 0
        A[1, 0], A[1, 1], A[1, 2] = -1, 0, 0

        #define lower elements
        q0 = q[0]
        qVec = q[1:]
        A[2, :] = 0.5 * qVec.T
        A[3:, :] = 0.5 * (q0 * ca.MX.eye(3) + self.hat_cas(qVec))

        #return matrix
        return A

    def eval_input(self, t):
        #get current state
        x = self.observer.get_state()
        q = x[2:]
        x = x[0: 2]

        #set up optimization problem
        opti = ca.Opti()
        uk = opti.variable(3, self.N - 1)
        qk = opti.variable(4, self.N)
        xk = opti.variable(2, self.N)

        #set up boundary conditions
        opti.subject_to(xk[:, 0] ==  x)
        opti.subject_to(qk[:, 0] == q)

        #set up x dynamics in terms of u
        for j in range(self.N-1):
            #extract vectors
            xj = xk[:, j]
            xjp1 = xk[:, j+1]
            qj = qk[:, j]
            qjp1 = qk[:, j+1]

            #extract input
            uj = uk[:, j]

            #get A matrix
            Aj = self.eval_A_cas(qj)

            #enforce constraint -> use A matrix to modify input
            opti.subject_to(ca.vertcat(xjp1, qjp1) == ca.vertcat(xj, qj) + self.h * Aj @ uj)

        #minimize a squared norm over the horizon
        termCost = (qk[:, -1] - self.qD).T @ self.QNq @ (qk[:, -1] - self.qD) + (xk[:, -1] - self.xD).T @ self.QNr @ (xk[:, -1] - self.xD)
        cost = termCost
        for i in range(self.N-1):
            cost = cost + (qk[:, i] - self.qD).T @ self.Qq @ (qk[:, i] - self.qD) + (xk[:, i] - self.xD).T @ self.Qr @ (xk[:, i] - self.xD) + uk[:, i].T @ self.R @ uk[:, i]

        #solve problem
        opti.minimize(cost)
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", option)
        sol = opti.solve()
        self._u = sol.value(uk)[:, 0].reshape((3, 1))
        # print(sol.value(cost))
        return self._u
