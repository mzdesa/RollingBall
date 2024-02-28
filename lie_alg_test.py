import numpy as np
import CalSim as cs
from scipy.linalg import logm, expm

class SimpleNH(cs.Dynamics):
    """
    Rolling Ball Nonholonomic dynamics
    """
    def __init__(self, x0):
        """
        Init function for rolling ball dynamics objet
        Inputs:
            x0 (9x1 NumPy Array): (x, xDot, R, omega) initial condition
        """
        #Define a desired rotation/position attribute (used for plotting in control)
        self.Rd = None
        self.rd = None

        #Define e3 basis vector and its hat map
        self.e3 = np.array([[0, 0, 1]]).T
        self.e3Hat = cs.hat(self.e3)
        self.I = np.eye(3)

        #define ball radius
        self.rho = 1

        #define A matrix
        self.A = np.array([[0, -1, 0], [1, 0, 0]])

        #Implement the state dynamics
        def f(x, u, t):
            """
            Dynamics function
            x = (r, rDot, R, omega)
            u = (f, M)
            """
            #extract components of state vector
            r, R = self.state_vec_2_tuple(x)

            #extract components of input vector
            omega = u

            #compute dynamics
            rDot = -self.rho * self.A @ omega
            rDot = np.vstack((rDot, 0))

            RDot = cs.hat(omega) @ R

            #Reshape rotation matrix into a vector
            RDot = (RDot.T).reshape((9, 1))
            
            #stack the dynamics terms and return xDot
            xDot = np.vstack((rDot, RDot))
            return xDot
        
        #Call the super init function -> default to one ball
        super().__init__(x0, 12, 3, f, N = 1)
        
    def state_vec_2_tuple(self, x):
        """
        Converts the state vector into the tuple (r, rDot, R, omega)
        Inputs:
            x (12x1 NumPy Array): state vector of system
        Reutns:
            r, R: Tuple containing vectors/matrices
        """
        #assemble the state vector
        r = x[0: 3, 0].reshape((3, 1))
        R = x[3:, 0].reshape((3, 3)).T

        #return tuple
        return r, R
    
class SimpleControl(cs.Controller):
    """
    Define an MPC snake controller
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

        #define hat of e3, projection
        self.e3 = np.array([[0, 0, 1]]).T
        self.e3Hat = cs.hat(self.e3)
        self.ProjPi = np.array([[1, 0, 0], [0, 1, 0]])

        #define current state
        x0 = self.observer.get_state()
        self.r0 = x0[0:3].reshape((3, 1))
        self.R0 = x0[3:, 0].reshape((3, 3)).T

        #define desired position and rotation
        self.rd = np.array([[1, 1, 0]]).T
        self.pd = self.ProjPi @ self.rd
        self.Rd = cs.calc_Rx(np.pi/4) @ cs.calc_Ry(np.pi/4)
        self.xid = self.rot_2_xi(self.Rd)

        #Set desired rotation in dynamics (for plotting purposes)
        self.observer.dynamics.Rd = self.Rd
        self.observer.dynamics.rd = self.rd

        #set trajectory parameters
        self.T = 10
        self.Delta = 5 #time of each linearization step -> set as control frequency
        self.B = -self.rho * np.array([[0, -1, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def state_vec_2_tuple(self, x):
        """
        Converts the state vector into the tuple (r, rDot, R, omega)
        Inputs:
            x (12x1 NumPy Array): state vector of system
        Reutns:
            r, R: Tuple containing vectors/matrices
        """
        #assemble the state vector
        r = x[0: 3, 0].reshape((3, 1))
        R = x[3:, 0].reshape((3, 3)).T

        #return tuple
        return r, R

    def rot_2_xi(self, R):
        """
        Convert a rotation matrix to xi coordinates
        """
        #take vee map of log of matrices
        return cs.vee_3d(logm(self.R0.T @ R))
    
    def calc_local_coords(self, r, R):
        """
        Takes in a state vector, computes local coordinates (p, xi)
        """
        #calculate p via projection
        p = r[0:2].reshape((2, 1))

        #calculate xi via log transform
        xi = self.rot_2_xi(R)

        #return result
        return np.vstack((p, xi))
    
    def update_init_state(self, x0):
        """
        Updates R0 for computation of local coordinates
        """
        #get r, R
        r, R = self.state_vec_2_tuple(x0)

        #update "initial" and goal states
        self.R0 = R

    def calc_W(self):
        """
        Compute gramian matrix
        """
        return self.B @ self.B.T * self.Delta

    def eval_input(self, t):
        """
        Calculate input omega
        """
        #get state
        x = self.observer.get_state()

        #update the local coordinates to be with respect to current state
        self.update_init_state(x)

        #calculate local coordinates for x0, x1 -> convert to (p, xi) coordinates
        loc_x0 = self.calc_local_coords(self.r0, self.R0)
        loc_xd = self.calc_local_coords(self.rd, self.Rd)

        #print error
        print(np.linalg.norm(loc_x0 - loc_xd))

        #calculate the iteration we are on
        self._u = -self.B.T @ np.linalg.pinv(self.calc_W()) @ (loc_x0 - loc_xd) 
        return self._u
    
if __name__ == "__main__":
    #create an object and run simulation
    r0 = 0 * np.ones((3, 1))
    R0 = (cs.calc_Rx(np.pi/2) @ cs.calc_Ry(np.pi/2)).T.reshape((9, 1)) #np.eye(3).reshape((9, 1))
    x0 = np.vstack((r0, R0))

    #define dynamics object
    ballDyn = SimpleNH(x0)

    #create an observer
    observerManager = cs.ObserverManager(ballDyn)

    #create a snake controller manager
    controllerManager = cs.ControllerManager(observerManager, SimpleControl)

    #create a ball environment
    env = cs.Environment(ballDyn, controllerManager, observerManager, T = 10)

    #run the simulation
    env.run()
