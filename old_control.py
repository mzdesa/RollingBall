import numpy as np
import CalSim as cs
import casadi as ca
from scipy.spatial.transform import *


class BallCLFNH(cs.Controller):
    """
    Define a position tracking CLF controller for the nonholonomic dynamics.
    """
    def __init__(self, observer, lyapunovBarrierList=None, trajectory=None, depthCam=None):
        """
        Init function for a min norm to full state linearization controller
        Inputs:
            observer (StateObserver): standard state observer
        """
        #first, call super init function on controller class
        super().__init__(observer, lyapunovBarrierList=None, trajectory=None, depthCam=None)

        #define desired position and rotation
        self.rd = np.array([[5, 6, 0]]).T
        # self.Rd = cs.calc_Rx(0.1) @ cs.calc_Ry(0.1)

        #define constants
        self.e3 = np.array([[0, 0, 1]]).T
        self.e3Hat = cs.hat(self.e3)
        self.I = np.eye(3)

        #radius
        self.rho = self.observer.dynamics.rho

        #clf constant
        self.gamma = 1
    
    def calc_r_term(self, r, rd, R):
        """
        Calculate positio term in error
        """
        return -self.rho * (r - rd).T @ self.e3Hat @ R
    
    def calc_V(self, r, rd):
        """
        Calculate value of Lyapunov function
        """
        return 0.5 * np.linalg.norm(r - rd)**2
    
    def get_des_state(self, t):
        """
        Returns desired state at time t
        """
        # r = 1
        # a = 1
        # rd = np.array([[r * np.cos(a*t) - r, r * np.sin(a*t), 0]]).T
        # rdDot = np.array([[-r*a*np.sin(a*t), r*a*np.cos(a*t), 0]]).T
        rd = self.rd
        rdDot = 0*self.rd
        return rd, rdDot

    def eval_input(self, t):
        """
        Evaluate the input to a snaks
        Inputs:
            t (float): current time in simulation
        """
        #get state
        x = self.observer.get_state()
        r = x[0: 3, 0].reshape((3, 1))
        R = x[3:, 0].reshape((3, 3)).T

        #get desired state
        rd, rdDot = self.get_des_state(t)

        #solve an optimization for u
        opti = ca.Opti()
        omega = opti.variable(3, 1)

        #define cost function as min norm to Gamma input
        cost = omega.T @ omega
        
        #define CLF constraint
        opti.subject_to(-(r - rd).T @ rdDot + (-self.rho * (r - rd).T @ self.e3Hat @ R) @ omega
                         <= -self.gamma * self.calc_V(r, self.rd))

        #set up optimizatoin
        opti.minimize(cost)
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", option)
        sol = opti.solve()
        
        #store the input
        self._u = sol.value(omega).reshape((3, 1))
        return self._u
    
    
class BallFBLinOrient(cs.Controller):
    """
    Define an orientation tracking feedback linearizing controller for the NH dynamics.
    """
    def __init__(self, observer, lyapunovBarrierList=None, trajectory=None, depthCam=None):
        """
        Init function for a min norm to full state linearization controller
        Inputs:
            observer (StateObserver): standard state observer
        """
        #first, call super init function on controller class
        super().__init__(observer, lyapunovBarrierList=None, trajectory=None, depthCam=None)

        #define desired position and rotation
        self.Rd = cs.calc_Rx(np.pi/2) @ cs.calc_Ry(np.pi/2) @ cs.calc_Rz(-np.pi/2)

        #define constants
        self.e3 = np.array([[0, 0, 1]]).T
        self.e3Hat = cs.hat(self.e3)
        self.I = np.eye(3)

        #radius
        self.rho = self.observer.dynamics.rho

        #gain matrix
        self.K = 1 * np.diag([1, 1, 1])

        #set a tolerance for zero
        self.tol = 1e-3

        #Set times for trajectory -> control freq. is 50 hz.
        self.times = np.arange(0, 10 + 1/50, 1/50).tolist()
        
        #define start & end quaternions and SLERP interpolation
        x0 = self.observer.get_state()
        R0 = x0[3:, 0].reshape((3, 3)).T
        self.slerp = Slerp([0, 10], Rotation.from_matrix([R0, self.Rd]))
        self.interp_rots = self.slerp(self.times).as_matrix()
    
    def get_des_orient(self, t):
        """
        Returns desired orientation at time t. Define a trajectory using the sim time.
        """
        index = int(t * 50)
        Rd = self.interp_rots[index, :, :]
        return Rd
    
    def get_xi_phi(self, R):
        """
        Calculates axis and angle corresponding to a rotation matrix R
        """
        phi = np.arccos((np.trace(R) - 1)/2)
        if phi <= self.tol:
            xi = np.zeros((3, 1))
        else:
            xi = 1/(2*np.sin(phi)) * np.array([[R[2, 1] - R[1, 2], R[0, 2] - R[2, 1], R[1, 0] - R[0, 1]]]).T
        return xi, phi

    def eval_input(self, t):
        """
        Evaluate the input.
        Inputs:
            t (float): current time in simulation
        """
        #get state
        x = self.observer.get_state()
        R = x[3:, 0].reshape((3, 3)).T

        #convert R to its axis-angle representation
        xi, phi = self.get_xi_phi(R)

        #get desired state and its axis-angle representation
        Rd = self.get_des_orient(t)
        xid, phid = self.get_xi_phi(Rd)

        #compute omega hat
        omegaHat = cs.hat(xi * phi) @ (self.K @ (cs.hat(xi * phi) - cs.hat(xid * phid)))
        omega = cs.vee_3d(omegaHat)
        
        #store the input
        self._u = omega
        return self._u
    

class BallCLFFull_Legacy(cs.Controller):
    """
    Define an orientation tracking feedback linearizing controller for the NH dynamics.
    """
    def __init__(self, observer, lyapunovBarrierList=None, trajectory=None, depthCam=None):
        """
        Init function for a min norm to full state linearization controller
        Inputs:
            observer (StateObserver): standard state observer
        """
        #first, call super init function on controller class
        super().__init__(observer, lyapunovBarrierList=None, trajectory=None, depthCam=None)

        #define desired position and rotation
        self.rd = np.array([[1, 1, 0]]).T
        self.Rd = cs.calc_Rx(np.pi/4) @ cs.calc_Ry(np.pi/4) @ cs.calc_Rz(-np.pi/2)

        #define constants
        self.e1 = np.array([[1, 0, 0]]).T
        self.e2 = np.array([[0, 1, 0]]).T
        self.e3 = np.array([[0, 0, 1]]).T
        self.ei = [self.e1, self.e2, self.e3]
        self.e3Hat = cs.hat(self.e3)
        self.I = np.eye(3)

        #radius
        self.rho = self.observer.dynamics.rho

        #gain matrix
        self.K = 1 * np.diag([1, 1, 1])

        #scaling in norms
        self.ki = [1, 1, 1, 2]

        #set CLF gamma
        self.gamma = 0.5

        #set tolerance on CLF constraint
        self.tol = 1e-4
    
    def get_des_orient(self, t):
        """
        Returns desired orientation at time t. Define a trajectory using the sim time.
        """
        return self.Rd
    
    def calc_psi_i(self, R, i):
        """
        Calculate error psi i.
        i = 0, 1, 2.
        """
        return 0.5*np.linalg.norm((self.Rd - R) @ self.ei[i])**2
    
    def calc_er(self, r):
        """
        Calculates CLF error to desired position.
        """
        return 0.5*np.linalg.norm(self.rd - r)
    
    def calc_psi_i_dot(self, R, omega, i):
        """
        Calculate psi_i time derivative.
        """
        return ((self.Rd - R) @ self.ei[i]).T @ R @ cs.hat(self.ei[i]) @ omega
    
    def calc_er_dot(self, r, R, omega):
        """
        Calculate time derivative of position error function
        """
        return self.rho * (self.rd - r).T @ self.e3Hat @ R @ omega
    
    def calc_v(self, r, R):
        """
        Calculate Lyapunov function V
        """
        V = 0
        for i in range(3):
            V = V + self.ki[i] * self.calc_psi_i(R, i)
        V = V + self.ki[3] * self.calc_er(r)
        return V
    
    def calc_v_dot(self, r, R, omega):
        """
        Returns derivative of V
        """
        vDot = 0
        for i in range(3):
            vDot = vDot + self.ki[i] * self.calc_psi_i_dot(R, omega, i)
        vDot = vDot + self.ki[3] * self.calc_er_dot(r, R, omega)
        return vDot

    def eval_input(self, t):
        """
        Evaluate the input.
        Inputs:
            t (float): current time in simulation
        """
        #get state
        x = self.observer.get_state()
        r = x[0:3].reshape((3, 1))
        R = x[3:, 0].reshape((3, 3)).T

        #create optimization for omega
        opti = ca.Opti()
        omega = opti.variable(3, 1)

        #enforce constraint
        V = self.calc_v(r, R)
        Vdot = self.calc_v_dot(r, R, omega)

        #if V meets tolerance cutoff, return.
        if V <= self.tol:
            self._u = np.zeros((3, 1))
            return self._u
        opti.subject_to(Vdot <= -self.gamma * V)

        #print value of Lyapunov function
        print(V)

        #define cost 
        cost = omega.T @ omega

        #perform minimization
        opti.minimize(cost)
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", option)
        sol = opti.solve()
        
        #store and return the input
        self._u = sol.value(omega).reshape((3, 1))
        return self._u
    

class BallCLFOrient(cs.Controller):
    """
    Define an orientation tracking CLF controller for the physical dynamics.
    May also track rotation when used with nonholonomic controller.
    """
    def __init__(self, observer, lyapunovBarrierList=None, trajectory=None, depthCam=None):
        """
        Init function for a min norm to full state linearization controller
        Inputs:
            observer (StateObserver): standard state observer
        """
        #first, call super init function on controller class
        super().__init__(observer, lyapunovBarrierList=None, trajectory=None, depthCam=None)

        #define desired position and rotation
        self.Rd = cs.calc_Rx(np.pi/4) @ cs.calc_Ry(np.pi/4) @ cs.calc_Rz(-np.pi/2)

        #define constants
        self.e1 = np.array([[1, 0, 0]]).T
        self.e2 = np.array([[0, 1, 0]]).T
        self.e3 = np.array([[0, 0, 1]]).T
        self.ei = [self.e1, self.e2, self.e3]
        self.e3Hat = cs.hat(self.e3)
        self.I = np.eye(3)

        #physical constants
        self.rho = self.observer.dynamics.rho
        self.m = self.observer.dynamics.m
        self.c = self.observer.dynamics.c

        #P and G matrices/functions
        self.P = self.observer.dynamics.P
        self.G = self.observer.dynamics.G

        #define alpha and epsilon CLF constants
        self.alpha = 1

        #state conversion functions
        self.state_vec_2_tuple = self.observer.dynamics.state_vec_2_tuple

        #set CLF gamma
        self.gamma = 5

        #Store the NH planning controller to generate the desired omega, set its Rd
        self.omegaController = BallCLFOrientNH(self.observer)
        self.omegaController.Rd = self.Rd

        #store the vector of ki
        self.ki = [1, 1, 1]
        self.KI = np.diag(self.ki)

    def get_des_state(self, t):
        """
        Returns desired rotation, angular velocity.
        """
        #return the omega input
        return self.Rd, np.zeros((3, 1))
    
    def calc_psi(self, R, Rd):
        """
        Calculate error psi i.
        i = 0, 1, 2.
        """
        psi = 0
        for i in range(3):
            psi = psi + self.ki[i] * (0.5*np.linalg.norm((Rd - R) @ self.ei[i])**2)
        return psi
    
    def calc_psi_dot(self, R, omega):
        """
        Calculate psi_i time derivative.
        """
        psiDot = 0
        for i in range(3):
            psiDot = psiDot + self.ki[i] * (((self.Rd - R) @ self.ei[i]).T @ R @ cs.hat(self.ei[i]) @ omega)
        return psiDot[0, 0]
    
    def calc_er(self, R, Rd):
        """
        Calculate orientation error vector er
        TODO: maybe replace these with the cross product terms?
        """
        er = (Rd - R) @ (self.KI @ np.ones((3, 1)))
        return er

    def calc_eomega(self, R, Rd, omega, omegaD):
        """
        Calculate derivative of error vector
        """
        return (Rd @ cs.hat(omegaD) - R @ cs.hat(omega)) @ (self.KI @ np.ones((3, 1)))
    
    def hat_ca(self, omega):
        """
        Computes hat map of a casadi matrix
        """
        omegaHat = ca.MX.zeros(3, 3)
        omegaHat[0, 0], omegaHat[0, 1], omegaHat[0, 2] = 0, -omega[2, 0], omega[1, 0]
        omegaHat[1, 0], omegaHat[1, 1], omegaHat[1, 2] = omega[2, 0], 0, -omega[0, 0]
        omegaHat[2, 0], omegaHat[2, 1], omegaHat[2, 2] = -omega[1, 0], omega[0, 0], 0
        return omegaHat
    
    def calc_eomega_dot(self, R, Rd, omega, omegaD, f, M):
        """
        Calculate time derivative of eOmega.
        """
        #calculate omegaDot
        omegaDot = 1/self.c * ((self.I + self.rho * R.T @ self.e3Hat @ self.P @ self.G(R)) @ M + self.rho * R.T @ self.e3Hat @ self.P @ f)
        omegaDDot = np.zeros((3, 1)) #assume zero omega desired derivative

        #calculate eOmegaDot
        return (Rd @ cs.hat(omegaD) @ cs.hat(omegaD) + Rd @ cs.hat(omegaDDot) - R @ cs.hat(omega) @ cs.hat(omega) - R @ self.hat_ca(omegaDot)) @ self.KI @ np.ones((3, 1))

    def calc_v(self, R, Rd, omega, omegaD):
        """
        Calculate Lyapunov function V
        """
        #calculate angular velocity error
        eOmega = self.calc_eomega(R, Rd, omega, omegaD)

        #calculate vector orientation error
        er = self.calc_er(R, Rd)

        #return Lyapunov function value
        # return 0.5 * eOmega.T @ eOmega + self.calc_psi(R, Rd) + self.alpha * eOmega.T @ er
        return 0.5 * eOmega.T @ eOmega + 0.5 * er.T @ er + self.alpha * eOmega.T @ er

    def calc_v_dot(self, R, Rd, omega, omegaD, f, M):
        """
        Returns derivative of V
        """
        eOmega = self.calc_eomega(R, Rd, omega, omegaD)
        eOmegaDot = self.calc_eomega_dot(R, Rd, omega, omegaD, f, M)
        er = self.calc_er(R, Rd)
        psiDot = self.calc_psi_dot(R, omega)

        #return derivative of V
        # return (eOmega + self.alpha * er).T @ eOmegaDot + psiDot + self.alpha * eOmega.T @ eOmega
        return (eOmega + self.alpha * er).T @ eOmegaDot + er.T @ eOmega + self.alpha * eOmega.T @ eOmega

    def disp_rotation_error(self, R, Rd):
        print("Rotation Error: ", np.trace(self.I - Rd.T @ R))

    def eval_input(self, t):
        """
        Evaluate the input.
        Inputs:
            t (float): current time in simulation
        """
        #get state
        x = self.observer.get_state()
        r, rDot, R, omega = self.state_vec_2_tuple(x)

        #get desired position, velocity
        Rd, omegaD = self.get_des_state(t)

        #create optimization for omega
        opti = ca.Opti()
        f = opti.variable(3, 1)
        M = opti.variable(3, 1)

        #enforce constraint
        V = self.calc_v(R, Rd, omega, omegaD)
        Vdot = self.calc_v_dot(R, Rd, omega, omegaD, f, M)
        opti.subject_to(Vdot <= -self.gamma * V)

        # print error to desired rotation
        self.disp_rotation_error(R, self.Rd)

        #define cost -> min norm over f, M
        cost = f.T @ f + M.T @ M

        #perform minimization
        opti.minimize(cost)
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", option)
        sol = opti.solve()
        
        #store and return the input
        f = sol.value(f).reshape((3, 1))
        M = sol.value(M).reshape((3, 1))
        self._u = np.vstack((f, M))
        return self._u