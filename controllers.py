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


class BallCLFOrient(cs.Controller):
    """
    Define an orientation tracking CLF controller for the physical dynamics.
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

    def get_des_omega(self, t):
        """
        Returns desired angular velocity at time t
        """
        #return the omega input
        return self.omegaController.eval_input(t)

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
    
    def calc_vomega(self, omega, omegaD):
        """
        Calculate Lyapunov function Vomega for angular velocity
        """
        #return value of V
        return (0.5 * (omega - omegaD).T @ (omega - omegaD))[0, 0]
    
    def calc_vomega_dot(self, omega, omegaD, R, f, M):
        """
        Returns derivative of Vomega
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

        #get desired position, angular velocity
        rD = self.rd
        vD = np.zeros((3, 1))
        omegaD = self.get_des_omega(t)

        #create optimization for omega
        opti = ca.Opti()
        f = opti.variable(3, 1)
        M = opti.variable(3, 1)

        #enforce constraint
        V = self.calc_vomega(omega, omegaD) + self.calc_vr(r, rD, rDot, vD)
        Vdot = self.calc_vomega_dot(omega, omegaD, R, f, M) + self.calc_vr_dot(r, rD, rDot, vD, R, f, M)
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