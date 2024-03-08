import numpy as np
import CalSim as cs
from dynamics_simulator import RollingBall

from scipy.spatial.transform import Rotation as Rsp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

class HamiltonianBall(cs.Dynamics):
    """
    Dynamics of the Hamiltonian system
    """
    def __init__(self, x0):
        """
        Init function for rolling ball dynamics objet
        Inputs:
            x0 (6x1 NumPy Array): (x, q) initial condition
            Note: q is a quaternion of the form (q0, qVec)
        """

        #Define e3 basis vector and its hat map
        self.I = np.eye(3)

        #define ball radius
        self.rho = 1

        #define A matrix as a funcntion of the quaternion
        self.A = np.array([[0, 1, 0], [-1, 0, 0]]) #position dynamics A

        #define initial values for lambdax1, lambdax2 (they are stationary)
        self.lambdax1 = 10
        self.lambdax2 = -1
        self.lambdax1x2 = np.array([[-self.lambdax2, self.lambdax1, 0]]).T

        #define time period T of hamiltonian
        self.T = 5

        #store an input history
        self.uData = np.zeros((3, 1))

        #Implement the state dynamics
        def f(x, u, t):
            """
            Dynamics function
            x = (x, r1, r2, r3, lambda1r, lambda2r, lambda3r)
            u = (omega)
            """
            #extract components of state vector
            r1 = x[2:5]
            r2 = x[5:8]
            r3 = x[8:11]
            lambdar1 = x[11:14]
            lambdar2 = x[14:17]
            lambdar3 = x[17:]

            #compute the input terms via the stationarity condition
            uStationary = -0.5 * (self.lambdax1x2 + cs.hat(r1) @ lambdar1 + cs.hat(r2) @ lambdar2 + cs.hat(r3) @ lambdar3)
            u = uStationary #np.vstack((uStationary[0:2], 0)) -> sets u to be zero
            # print(np.linalg.norm(u))
            self.uData = np.hstack((self.uData, u))
            uHat = cs.hat(u)

            #translational dynamics
            XDot = self.A @ u

            #rotational dynamics
            r1Dot = uHat @ r1
            r2Dot = uHat @ r2
            r3Dot = uHat @ r3

            #multiplier dynamics
            lambdar1Dot = uHat @ lambdar1
            lambdar2Dot = uHat @ lambdar2
            lambdar3Dot = uHat @ lambdar3

            #stack vectors
            xDot = np.vstack((XDot, r1Dot, r2Dot, r3Dot, lambdar1Dot, lambdar2Dot, lambdar3Dot))
            return xDot
        
        #Call the super init function -> default to one ball
        super().__init__(x0, 20, 0, f, N = 1)

    def H(self, x):
        """
        Compute Hamiltonian Function (used in plotting)
        """
        #extract components of state vector
        r1 = x[2:5]
        r2 = x[5:8]
        r3 = x[8:11]
        lambdar1 = x[11:14]
        lambdar2 = x[14:17]
        lambdar3 = x[17:]
        
        #compute the associated input
        u = -0.5 * (self.lambdax1x2 + cs.hat(r1) @ lambdar1 + cs.hat(r2) @ lambdar2 + cs.hat(r3) @ lambdar3)
        u1, u2 = u[0, 0], u[1, 0]

        #compute Hamiltonian
        hamilt = u.T @ u + self.lambdax1 * u2 - self.lambdax2 * u1 + -(lambdar1.T @ cs.hat(r1) + lambdar2.T @ cs.hat(r2) + lambdar3.T @ cs.hat(r3)) @ u
        return hamilt[0, 0]

    def gen_sphere(self, R, r):
        """
        Generate mesh for sphere of radius rho at the origin with orientation R and position r
        """
        return RollingBall.gen_sphere(self, R, r)
    
    def show_plots(self, xData, uData, tData, stateLabels=None, inputLabels=None, obsManager=None):
        #call super plots function
        super().show_plots(xData, uData, tData, stateLabels, inputLabels, obsManager)

        #plot the stored inputs from the Hamiltonian system
        uData = self.uData[:, 1:] #slice off first default entry

        #extract lists
        u1 = uData[0, :].tolist()
        u2 = uData[1, :].tolist()
        u3 = uData[2, :].tolist()

        #compute time list
        time = np.linspace(0, self.T, len(u1))

        #plot results
        plt.plot(time, u1)
        plt.plot(time, u2)
        plt.plot(time, u3)
        plt.legend(["u1", "u2", "u3"])
        plt.title("Evolution of Inputs to System")
        plt.show()

        #compute Hamiltonian
        Hhist = []
        for i in range(xData.shape[1]):
            #compute H
            Hhist.append(self.H(xData[:, i]))

        #plot out Hamiltonian
        plt.plot(tData[0, :], Hhist)
        plt.show()

    def show_animation(self, xData, uData, tData, animate = True, obsManager = None):
        #Set constant animtion parameters
        FREQ = 50 #control frequency, same as data update frequency
        
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')

        #set axis min max limits
        x0Hist = xData[0, :].tolist()
        xMin, xMax = np.min(x0Hist), np.max(x0Hist)
        xMin -= self.rho
        xMax += self.rho
        y0Hist = xData[1, :].tolist()
        yMin, yMax = np.min(y0Hist), np.max(y0Hist)
        yMin -= self.rho
        yMax += self.rho
        zMax = 5

        def update(idx):
            ax.cla()

            #set axis aspect ratio & limits
            ax.set_box_aspect((1, (yMax - yMin)/(xMax - xMin), zMax/(xMax - xMin)))
            ax.set_xlim(xMin, xMax)
            ax.set_ylim(yMin, yMax)
            ax.set_zlim(0, zMax)

            #set title
            ax.set_title("Trajectory of Ball")

            #get state and convert to r, rDot, R, omega
            state = xData[:, idx].reshape((self.singleStateDimn, 1))
            x = state[0:2]
            r1 = state[2:5]
            r2 = state[5:8]
            r3 = state[8:11]

            #convert ri to a rotation
            R = np.hstack((r1, r2, r3))

            #generate sphere -> append a zero to the position for z
            x, y, z = self.gen_sphere(R, np.vstack((x, 0)))

            #plot the 3D surface on the axes
            ax.plot_surface(x, y, z, cmap=plt.cm.viridis, linewidth=0.1)

            return fig,
    
        num_frames = xData.shape[1]-1
        anim = animation.FuncAnimation(fig = fig, func = update, frames=num_frames, interval=1/FREQ*1000)
        plt.show()

if __name__ == '__main__':
    #Run a test of the Hamiltonian system
    r0 = 0 * np.ones((2, 1))
    R0 = np.eye(3)
    r1, r2, r3 = R0[:, 0].reshape((3, 1)), R0[:, 1].reshape((3, 1)), R0[:, 2].reshape((3, 1))
    lambda0 = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]]).T
    x0 = np.vstack((r0, r1, r2, r3, lambda0))

    #define dynamics object
    ballDyn = HamiltonianBall(x0)

    #create an observer
    observerManager = cs.ObserverManager(ballDyn)

    #create a snake controller manager
    controllerManager = None

    #create a ball environment
    env = cs.Environment(ballDyn, controllerManager, observerManager, T = 10)

    #run the simulation
    env.run()