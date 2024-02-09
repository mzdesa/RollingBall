"""
Define a controller for the physical ball dynamics
"""
import numpy as np
import CalSim as cs
import casadi as ca
from scipy.spatial.transform import *


class BallCLFPos(cs.Controller):
    """
    Define an position tracking CLF controller for the physical dynamics.
    """
    def __init__(self, observer, lyapunovBarrierList=None, trajectory=None, depthCam=None):
        """
        Init function for a position-tracking CLF controller
        Inputs:
            observer (StateObserver): standard state observer
        """
        #first, call super init function on controller class
        super().__init__(observer, lyapunovBarrierList=None, trajectory=None, depthCam=None)

        #define desired position and rotation
        self.rd = np.array([[1, 1, 0]]).T

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
        self.epsilon = np.sqrt(self.alpha)

        #state conversion functions
        self.state_vec_2_tuple = self.observer.dynamics.state_vec_2_tuple

        #set CLF gamma
        self.gamma = 2

    def get_des_state(self, t):
        """
        Returns desired state at time t
        """
        r = 2
        a = 2
        rd = np.array([[r * np.cos(a*t) - r, r * np.sin(a*t), 0]]).T
        rdDot = np.array([[-r*a*np.sin(a*t), r*a*np.cos(a*t), 0]]).T
        return rd, rdDot
    
    def calc_v(self, r, rD, v, vD):
        """
        Calculate Lyapunov function V
        """
        #compute position and velocity error
        er = r - rD
        ev = v - vD

        #return value of V
        return (0.5 * ev.T @ ev + self.alpha/2 * er.T @ er + self.epsilon * er.T @ ev)[0, 0]
    
    def calc_v_dot(self, r, rD, v, vD, R, f, M):
        """
        Returns derivative of V
        """
        #compute position and velocity error
        er = r - rD
        ev = v - vD

        #compute derivative of velocity error s.t. dynamics
        rDDotD = np.zeros((3, 1))
        evDot = 1/self.m * (-self.P @ self.G(R) @ M + (np.eye(3) - self.P) @ f) - rDDotD

        #return value of VDot
        return (ev + self.epsilon * er).T @ evDot + (self.alpha * er + ev).T @ ev
        
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
        rD, vD = self.get_des_state(t)

        #create optimization for omega
        opti = ca.Opti()
        f = opti.variable(3, 1)
        M = opti.variable(3, 1)

        #enforce constraint
        V = self.calc_v(r, rD, rDot, vD)
        Vdot = self.calc_v_dot(r, rD, rDot, vD, R, f, M)
        opti.subject_to(Vdot <= -self.gamma * V)

        # #print value of Lyapunov function
        # print(V)

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


class BallCLFOmega(cs.Controller):
    """
    Define an angular velocity tracking CLF controller for the physical dynamics.
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
        self.epsilon = np.sqrt(self.alpha)

        #state conversion functions
        self.state_vec_2_tuple = self.observer.dynamics.state_vec_2_tuple

        #set CLF gamma
        self.gamma = 2

        #Store the NH planning controller to generate the desired omega, set its Rd
        self.omegaController = BallCLFOrientNH(self.observer)
        self.omegaController.Rd = self.Rd

    def get_des_state(self, t):
        """
        Returns desired angular velocity at time t
        """
        #return the omega input
        return self.omegaController.eval_input(t)
    
    def calc_v(self, omega, omegaD):
        """
        Calculate Lyapunov function V
        """
        #return value of V
        return (0.5 * (omega - omegaD).T @ (omega - omegaD))[0, 0]
    
    def calc_v_dot(self, omega, omegaD, R, f, M):
        """
        Returns derivative of V
        """
        #return value of VDot
        return 1/self.c * (omega - omegaD).T @ ((self.I + self.rho * R.T @ self.e3Hat @ self.P @ self.G(R)) @ M + self.rho * R.T @ self.e3Hat @ self.P @ f)

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
        omegaD = self.get_des_state(t)

        #create optimization for omega
        opti = ca.Opti()
        f = opti.variable(3, 1)
        M = opti.variable(3, 1)

        #enforce constraint
        V = self.calc_v(omega, omegaD)
        Vdot = self.calc_v_dot(omega, omegaD, R, f, M)
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

    
class BallCLFOrientNH(cs.Controller):
    """
    Define an orientation tracking CLF controller for the NH dynamics.
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

        #radius
        self.rho = self.observer.dynamics.rho

        #gain matrix
        self.K = 1 * np.diag([1, 1, 1])

        #scaling in norms
        self.ki = [1, 2, 3]

        #set CLF gamma
        self.gamma = 1

        #set tolerance on CLF constraint
        self.tol = 1e-4

        #state conversion functions
        self.state_vec_2_tuple = self.observer.dynamics.state_vec_2_tuple
    
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
    
    def calc_psi_i_dot(self, R, omega, i):
        """
        Calculate psi_i time derivative.
        """
        return ((self.Rd - R) @ self.ei[i]).T @ R @ cs.hat(self.ei[i]) @ omega
    
    def calc_v(self, R):
        """
        Calculate Lyapunov function V
        """
        V = 0
        for i in range(3):
            V = V + self.ki[i] * self.calc_psi_i(R, i)
        return V
    
    def calc_v_dot(self, R, omega):
        """
        Returns derivative of V
        """
        vDot = 0
        for i in range(3):
            vDot = vDot + self.ki[i] * self.calc_psi_i_dot(R, omega, i)
        return vDot

    def eval_input(self, t):
        """
        Evaluate the input.
        Inputs:
            t (float): current time in simulation
        """
        #get state
        x = self.observer.get_state()
        r, rDot, R, omega = self.state_vec_2_tuple(x)

        #create optimization for omega
        opti = ca.Opti()
        omega = opti.variable(3, 1)

        #enforce constraint
        V = self.calc_v(R)
        Vdot = self.calc_v_dot(R, omega)

        #if V meets tolerance cutoff, return.
        if V <= self.tol:
            self._u = np.zeros((3, 1))
            return self._u
        opti.subject_to(Vdot <= -self.gamma * V)

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
        self.gamma = 1

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
        return 0.5*np.trace(self.I - Rd.T @ R)
        
    def calc_psi_dot(self, R, Rd, omega):
        """
        Calculate psi_i time derivative.
        """
        return self.calc_eR(R, Rd).T @ omega
    
    def calc_eR(self, R, Rd):
        """
        Calculate orientation error vector er
        TODO: maybe replace these with the cross product terms?
        """
        return 0.5 * cs.vee_3d(Rd.T @ R - R.T @ Rd)

    def calc_eomega(self, R, Rd, omega, omegaD):
        """
        Calculate derivative of error vector
        """
        return omega - R.T @ Rd @ omegaD
    
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
        return omegaDot

    def calc_v(self, R, Rd, omega, omegaD):
        """
        Calculate Lyapunov function V
        """
        #calculate angular velocity error
        eOmega = self.calc_eomega(R, Rd, omega, omegaD)

        #calculate vector orientation error
        er = self.calc_eR(R, Rd)

        #return Lyapunov function value
        return 0.5 * eOmega.T @ eOmega + self.calc_psi(R, Rd) + self.alpha * eOmega.T @ er

    def calc_v_dot(self, R, Rd, omega, omegaD, f, M):
        """
        Returns derivative of V
        """
        eOmega = self.calc_eomega(R, Rd, omega, omegaD)
        eOmegaDot = self.calc_eomega_dot(R, Rd, omega, omegaD, f, M)
        eR = self.calc_eR(R, Rd)
        psiDot = self.calc_psi_dot(R, Rd, omega)

        #return derivative of V
        return (eOmega + self.alpha * eR).T @ eOmegaDot + psiDot + self.alpha * eOmega.T @ eOmega

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


class BallCLF(cs.Controller):
    """
    Define a position + orientation tracking CLF controller for the physical dynamics.
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
        self.rd = np.array([[1, 0, 0]]).T
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
        self.epsilon = np.sqrt(self.alpha)

        #state conversion functions
        self.state_vec_2_tuple = self.observer.dynamics.state_vec_2_tuple

        #set CLF gamma
        self.gamma = 1

        #Set desired rotation in dynamics (for plotting purposes)
        self.observer.dynamics.Rd = self.Rd
        self.observer.dynamics.rd = self.rd

    def get_des_state(self, t):
        """
        Returns desired position, velocity, rotation, angular velocity.
        """
        #return the omega input
        return self.rd, np.zeros((3, 1)), self.Rd, np.zeros((3, 1))

    def calc_vr(self, r, rD, v, vD):
        """
        Calculate Lyapunov function Vr for position
        """
        #compute position and velocity error
        er = r - rD
        ev = v - vD

        #return value of V
        return (0.5 * ev.T @ ev + self.alpha/2 * er.T @ er + self.epsilon * er.T @ ev)[0, 0]
    
    def calc_vr_dot(self, r, rD, v, vD, R, f, M):
        """
        Returns derivative of Vr
        """
        #compute position and velocity error
        er = r - rD
        ev = v - vD

        #compute derivative of velocity error s.t. dynamics
        rDDotD = np.zeros((3, 1))
        evDot = 1/self.m * (-self.P @ self.G(R) @ M + (np.eye(3) - self.P) @ f) - rDDotD

        #return value of VDot
        return (ev + self.epsilon * er).T @ evDot + (self.alpha * er + ev).T @ ev
    
    def calc_psi(self, R, Rd):
        """
        Calculate error psi i.
        i = 0, 1, 2.
        """
        return 0.5*np.trace(self.I - Rd.T @ R)
        
    def calc_psi_dot(self, R, Rd, omega):
        """
        Calculate psi_i time derivative.
        """
        return self.calc_eR(R, Rd).T @ omega
    
    def calc_eR(self, R, Rd):
        """
        Calculate orientation error vector er
        TODO: maybe replace these with the cross product terms?
        """
        return 0.5 * cs.vee_3d(Rd.T @ R - R.T @ Rd)

    def calc_eomega(self, R, Rd, omega, omegaD):
        """
        Calculate derivative of error vector
        """
        return omega - R.T @ Rd @ omegaD
    
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
        return omegaDot

    def calc_vR(self, R, Rd, omega, omegaD):
        """
        Calculate Lyapunov function V
        """
        #calculate angular velocity error
        eOmega = self.calc_eomega(R, Rd, omega, omegaD)

        #calculate vector orientation error
        er = self.calc_eR(R, Rd)

        #return Lyapunov function value
        return 0.5 * eOmega.T @ eOmega + self.calc_psi(R, Rd) + self.alpha * eOmega.T @ er

    def calc_vR_dot(self, R, Rd, omega, omegaD, f, M):
        """
        Returns derivative of V
        """
        eOmega = self.calc_eomega(R, Rd, omega, omegaD)
        eOmegaDot = self.calc_eomega_dot(R, Rd, omega, omegaD, f, M)
        eR = self.calc_eR(R, Rd)
        psiDot = self.calc_psi_dot(R, Rd, omega)

        #return derivative of V
        return (eOmega + self.alpha * eR).T @ eOmegaDot + psiDot + self.alpha * eOmega.T @ eOmega

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

        #get desired position, angular velocity
        rD, vD, Rd, omegaD = self.get_des_state(t)

        #create optimization for omega
        opti = ca.Opti()
        f = opti.variable(3, 1)
        M = opti.variable(3, 1)
        gamma = opti.variable()

        #enforce constraint
        V = self.calc_vR(R, Rd, omega, omegaD) + self.calc_vr(r, rD, rDot, vD)
        Vdot = self.calc_vR_dot(R, Rd, omega, omegaD, f, M) + self.calc_vr_dot(r, rD, rDot, vD, R, f, M)
        opti.subject_to(Vdot <= -gamma * V)
        # opti.subject_to(gamma >= 0.1)
        opti.subject_to(gamma == 10)

        # print error to desired rotation
        # self.disp_rotation_error(R, self.Rd)

        #define cost -> min norm over f, M
        cost = f.T @ f + M.T @ M - gamma

        #perform minimization
        opti.minimize(cost)
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", option)
        sol = opti.solve()

        # #check matrix rank
        # Rank_Matr1 = np.hstack((1/self.m * (self.I - self.P), -1/self.m * self.P @ self.G(R)))
        # Rank_Matr2 = np.hstack((1/self.c * self.rho * R.T @ self.e3Hat @ self.P, 1/self.c * (self.I + self.rho * R.T @ self.e3Hat @ self.P @ self.G(R))))
        # Rank_Matr = np.vstack((Rank_Matr1, Rank_Matr2))
        # if (np.linalg.matrix_rank(Rank_Matr) != 5):
        #     print("RANK: ", np.linalg.matrix_rank(Rank_Matr))
        #     print(np.linalg.pinv(Rank_Matr) @ Rank_Matr)

        #store and return the input
        f = sol.value(f).reshape((3, 1))
        M = sol.value(M).reshape((3, 1))
        # print(sol.value(gamma))
        self._u = np.vstack((f, M))
        return self._u
    

class BallFbLin(cs.Controller):
    """
    Define an angular velocity tracking CLF controller for the physical dynamics.
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
        self.rd = np.array([[0, 0, 0]]).T
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

        #state conversion functions
        self.state_vec_2_tuple = self.observer.dynamics.state_vec_2_tuple

        #Store the NH planning controller to generate the desired omega, set its Rd
        self.omegaController = BallCLFOrientNH(self.observer)
        self.omegaController.Rd = self.Rd

        #Define gain matrices
        self.Kp = 4*np.diag([1, 1, 1])
        self.Kd = 4*np.diag([1, 1, 1])

        #Define cutoff matrix
        self.A = np.array([[1, 0], [0, 1], [0, 0]])

        #Set desired rotation in dynamics (for plotting purposes)
        self.observer.dynamics.Rd = self.Rd
        self.observer.dynamics.rd = self.rd

    def get_des_state(self, t):
        """
        Returns desired position, velocity, and angular velocity at time t
        """
        #return the omega input
        return self.rd, np.zeros((3, 1)), self.omegaController.eval_input(t)
    
    def calc_K(self, R, reduced = False):
        """
        Calculate K term in dynamics
        """
        #form blocks in the matrix
        k1 = self.A.T @ (1/self.m * (self.I - self.P)) @ self.A
        k2 = self.A.T @ (-1/self.m * self.P @ self.G(R))
        k3 = (1/self.c * self.rho * R.T @ self.e3Hat @ self.P) @ self.A
        k4 = 1/self.c * (self.I + self.rho * R.T @ self.e3Hat @ self.P @ self.G(R))

        #Assemble the reduced order matrix
        Kred = np.vstack((np.hstack((np.eye(2), k2 @ np.linalg.inv(k4))), 
                          np.hstack((k3 @ np.linalg.inv(k1), np.eye(3)))))
        if np.linalg.matrix_rank(Kred) != 5:
            print(np.linalg.matrix_rank(Kred))

        #check matrix rank
        if not reduced:
            K1 = np.hstack((k1, k2))
            K2 = np.hstack((k3, k4))
            return np.vstack((K1, K2))
        else:
            return k1, k2, k3, k4, Kred
    
    def eval_nu(self, t, r, rDot, R, omega, rd, vd, omegad):
        """
        Calculate nu1, nu2
        """
        k1, k2, k3, k4, Kred = self.calc_K(R, reduced = True)

        v1 = self.A.T @ (self.Kp @ (rd - r) + self.Kd @ (vd - rDot))
        v2 = self.Kp @ (omegad - omega)

        nu = np.linalg.pinv(Kred) @ (np.vstack((v1, v2)))
        return nu[0:2], nu[2:]
        
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
        rd, vd, omegaD = self.get_des_state(t)

        #compute K matrix
        k1, k2, k3, k4, Kred = self.calc_K(R, reduced = True)
        nu1, nu2 = self.eval_nu(t, r, rDot, R, omega, rd, vd, omegaD)
        fPrime = np.linalg.inv(k1) @ nu1
        f = np.vstack((fPrime, 0))
        M = np.linalg.inv(k4) @ nu2
        u = np.vstack((f, M))
        if np.linalg.norm(u) >= 100:
            print("Pinv Error")
            self._u = np.random.rand(6, 1)
            return self._u
        self._u = u
        return self._u