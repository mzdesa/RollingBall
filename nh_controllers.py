"""
Define a controller for the snake
"""
import numpy as np
import CalSim as cs
import casadi as ca
from scipy.spatial.transform import *


class BallFBLinPos(cs.Controller):
    """
    Define a position tracking feedback linearizing controller for the NH dynamics.
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

        #define constants
        self.e3 = np.array([[0, 0, 1]]).T
        self.e3Hat = cs.hat(self.e3)
        self.I = np.eye(3)

        #radius
        self.rho = self.observer.dynamics.rho

        #gain matrix
        self.K = np.diag([1, 1, 0])
    
    def get_des_state(self, t):
        """
        Returns desired state at time t
        """
        r = 2
        a = 2
        rd = np.array([[r * np.cos(a*t) - r, r * np.sin(a*t), 0]]).T
        rdDot = np.array([[-r*a*np.sin(a*t), r*a*np.cos(a*t), 0]]).T
        # rd = self.rd
        # rdDot = 0*self.rd
        return rd, rdDot

    def eval_input(self, t):
        """
        Evaluate the input.
        Inputs:
            t (float): current time in simulation
        """
        #get state
        x = self.observer.get_state()
        r = x[0: 3, 0].reshape((3, 1))
        R = x[3:, 0].reshape((3, 3)).T

        #get desired state
        rd, rdDot = self.get_des_state(t)

        #solve for omega
        omega = -1/self.rho * R.T @ self.e3Hat @ (self.K @ (r - rd) + rdDot)
        
        #store the input
        self._u = omega
        return self._u

class BallFBLinPos(cs.Controller):
    """
    Define a position tracking feedback linearizing controller for the NH dynamics.
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

        #define constants
        self.e3 = np.array([[0, 0, 1]]).T
        self.e3Hat = cs.hat(self.e3)
        self.I = np.eye(3)

        #radius
        self.rho = self.observer.dynamics.rho

        #gain matrix
        self.K = np.diag([1, 1, 0])
    
    def get_des_state(self, t):
        """
        Returns desired state at time t
        """
        r = 2
        a = 2
        rd = np.array([[r * np.cos(a*t) - r, r * np.sin(a*t), 0]]).T
        rdDot = np.array([[-r*a*np.sin(a*t), r*a*np.cos(a*t), 0]]).T
        # rd = self.rd
        # rdDot = 0*self.rd
        return rd, rdDot

    def eval_input(self, t):
        """
        Evaluate the input.
        Inputs:
            t (float): current time in simulation
        """
        #get state
        x = self.observer.get_state()
        r = x[0: 3, 0].reshape((3, 1))
        R = x[3:, 0].reshape((3, 3)).T

        #get desired state
        rd, rdDot = self.get_des_state(t)

        #solve for omega
        omega = -1/self.rho * R.T @ self.e3Hat @ (self.K @ (r - rd) + rdDot)
        
        #store the input
        self._u = omega
        return self._u
    
class BallCLFOrient(cs.Controller):
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
        R = x[3:, 0].reshape((3, 3)).T

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
    
class BallCLF(BallCLFOrient):
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
        self.Rd = cs.calc_Rx(np.pi/4) @ cs.calc_Ry(np.pi/4)

        #scaling in norms
        self.ki = [1, 1, 1]

        #set tuning terms
        self.gamma = 1
        self.pr = 2 #tuning for relaxation
        self.pR = 1 

        #set tolerance on CLF constraint
        self.tol = 1e-4

        #define controller for "nominal" position tracking input
        self.posControl = BallFBLinPos(observer, lyapunovBarrierList, trajectory, depthCam)

    def calc_er(self, r):
        """
        Calculates CLF error to desired position.
        """
        return 0.5*np.linalg.norm(self.rd - r)
    
    def calc_er_dot(self, r, R, omega):
        """
        Calculate time derivative of position error function
        """
        return self.rho * (self.rd - r).T @ self.e3Hat @ R @ omega

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

        #define relaxation variables for R and r
        deltaR = opti.variable()
        deltar = opti.variable()

        #Get CLF for orientation
        VR = self.calc_v(R)
        VRDot = self.calc_v_dot(R, omega)
        print("VR: ", VR)

        #Get CLF for position
        Vr = self.calc_er(r)        
        VrDot = self.calc_er_dot(r, R, omega)
        print("Vr: ", Vr)

        #if V meets tolerance cutoff, return.
        if VR <= self.tol:
            self._u = np.zeros((3, 1))
            return self._u
        opti.subject_to(VRDot <= -self.gamma * VR + deltaR)
        opti.subject_to(VrDot <= -self.gamma * Vr + deltar)

        #define cost
        cost = omega.T @ omega + self.pR * deltaR**2 + self.pr * deltar**2

        #perform minimization
        opti.minimize(cost)
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", option)
        sol = opti.solve()
        
        #store and return the input
        self._u = sol.value(omega).reshape((3, 1))
        return self._u
    

class EulerPlanner(cs.Controller):
    """
    Define position & orientation planner for the NH dynamics.
    """
    def __init__(self, observer, lyapunovBarrierList=None, trajectory=None, depthCam=None):
        """
        Init function for a min norm to full state linearization controller
        Inputs:
            observer (StateObserver): standard state observer
        """
        #first, call super init function on controller class
        super().__init__(observer, lyapunovBarrierList=None, trajectory=None, depthCam=None)

        #get radius of sphere
        self.rho = self.observer.dynamics.rho

        #define desired position and rotation
        self.rd = np.array([[1, 1, 0]]).T
        self.Rd = cs.calc_Rx(np.pi/4) @ cs.calc_Ry(np.pi/4)

        #define current state
        x0 = self.observer.get_state()
        self.r0 = x0[0:3].reshape((3, 1))
        self.R0 = x0[3:, 0].reshape((3, 3)).T

        #Set trajectory length - Total length and times to roll along each axis
        self.T = 5
        self.TRx = 3.5/2 #time to roll along x axis
        self.TRy = 3.5/2 #time to roll along y axis
        self.Tz = 1/2 #time to spin around a z axis

        #define and calculate theta angles
        self.theta1 = 0
        self.theta2 = 0
        self.omegaY = self.calc_y()
        self.omegaX = self.calc_x()
        self.gammaI = []
        self.calc_gamma()
        self.omegaZ1 = self.calc_rz(0)
        self.omegaZ2 = self.calc_rz(1)
        self.omegaZ3 = self.calc_rz(2)

    def calc_y(self):
        """
        Function to calculate rotation about y axis
        """
        #first, define y axis
        omega = np.array([[0, 1, 0]]).T

        #calculate delta
        deltaX = self.rd[0, 0] - self.r0[0, 0]

        #calculate angle we need to turn. Positive angle moves right.
        self.theta1 = deltaX/self.rho

        #divide by rotation about the y axis
        omegaNorm = self.theta1/self.TRy

        #return scaled omega
        return omegaNorm * omega
    
    def calc_x(self):
        """
        Function to calculate rotation about y axis
        """
        #first, define y axis
        omega = np.array([[1, 0, 0]]).T

        #calculate delta
        deltaY = self.rd[1, 0] - self.r0[1, 0]

        #calculate angle we need to turn. Positive angle moves down.
        self.theta2 = -deltaY/self.rho

        #divide by rotation about the y axis
        omegaNorm = self.theta2/self.TRx

        #return scaled omega
        return omegaNorm * omega
    
    def calc_R(self, gamma1, gamma2, gamma3, theta1, theta2):
        """
        Calculate the full rotation matrix given gamma_i, theta_i
        """
        return (cs.calc_Rz(gamma3)
                @ cs.calc_Rx(theta2)
                @ cs.calc_Rz(gamma2)
                @ cs.calc_Ry(theta1)
                @ cs.calc_Rz(gamma1) @ self.R0)
    
    def calc_gamma(self):
        """
        Function to calculate gamma_i variables for pivots
        """
        #solve for the gamma variables
        opti = ca.Opti()
        gamma1 = opti.variable()
        gamma2 = opti.variable()
        gamma3 = opti.variable()

        #set up cost function and solve optimization
        cost = ca.trace(np.eye(3) - self.Rd.T @ self.calc_R(gamma1, gamma2, gamma3, self.theta1, self.theta2))
        opti.minimize(cost)
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", option)
        sol = opti.solve()

        self.CALC_R = self.calc_R(sol.value(gamma1), sol.value(gamma2), sol.value(gamma3), self.theta1, self.theta2)

        #store solutions variables
        self.gammaI = [sol.value(gamma1), sol.value(gamma2), sol.value(gamma3)]

    def calc_rz(self, i):
        """
        Calculates the angular velocity vector omega_i for gamma_i
        """
        #return scaled omega
        return self.gammaI[i]/self.Tz * np.array([[0, 0, 1]]).T
    
    def eval_input(self, t):
        """
        Calculate input
        """
        #define current state
        x = self.observer.get_state()
        r = x[0:3].reshape((3, 1))
        R = x[3:, 0].reshape((3, 3)).T

        if t <= self.Tz:
            #Apply the first gamma rotation
            self._u = R.T @ self.omegaZ1
        elif t <= self.TRy + self.Tz:
            #Apply the rotation in the spatial y direction
            self._u = R.T @ self.omegaY

        elif t <= self.TRy + 2*self.Tz:
            #Apply the second gamma rotation
            self._u = R.T @ self.omegaZ2

        elif t <= self.TRy + 2*self.Tz + self.TRx:
            #Apply the rotation in the spatial x direction
            self._u = R.T @ self.omegaX
        
        elif t <= self.T:
            #Apply the third gamma rotation
            self._u = R.T @ self.omegaZ3

        else:
            self._u = np.zeros((3, 1))

        if abs(t - self.T) <= 1e-4:
            print("***********************")
            print("Final Orientation Error: ", np.linalg.norm(R - self.Rd))
            print("***********************")

        #return the input
        return self._u