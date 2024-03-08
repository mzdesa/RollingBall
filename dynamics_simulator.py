import numpy as np
import CalSim as cs
from scipy.spatial.transform import Rotation as Rsp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation


class RollingBall(cs.Dynamics):
    """
    Rolling Ball dynamics
    """
    def __init__(self, r0, R0, omega0):
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

        #physical constants
        self.m = 1 # mass (kg)
        self.rho = 1 # radius (m)
        self.c = 2/5 * self.m * self.rho**2 # intertia

        #Define P(R) and G(R)
        self.P = np.linalg.inv(np.eye(3) + self.m*self.rho**2/self.c* self.e3Hat @ self.e3Hat.T)
        self.G = lambda R: self.m*self.rho/self.c * self.e3Hat @ R

        def comp_x0(r0, R0, omega0):
            """
            Computes an initial condition for the system based on the NH constraint.
            """
            rDot0 = self.rho * cs.hat(R0.reshape((3, 3)).T @ omega0) @ np.array([[0, 0, 1]]).T
            return np.vstack((r0, rDot0, R0, omega0))

        #Implement the state dynamics
        def f(x, u, t):
            """
            Dynamics function
            x = (r, rDot, R, omega)
            u = (f, M)
            """
            #extract components of state vector
            r, rDot, R, omega = self.state_vec_2_tuple(x)

            #extract components of input vector
            f, M = self.input_vec_2_tuple(u)

            #compute the dynamics terms
            rDDot = 1/self.m * (-self.P @ self.G(R) @ M + (self.I - self.P) @ f)
            RDot = R @ cs.hat(omega)
            omegaDot = 1/self.c * self.rho * R.T @ self.e3Hat @ self.P @ f + 1/self.c*(np.eye(3) + self.rho*R.T @ self.e3Hat @ self.P @ self.G(R)) @ M

            #Reshape rotation matrix into a vector
            RDot = (RDot.T).reshape((9, 1))
            
            #stack the dynamics terms and return xDot
            xDot = np.vstack((rDot, rDDot, RDot, omegaDot))
            return xDot
        
        #Call the super init function -> default to one ball
        super().__init__(comp_x0(r0, R0, omega0), 18, 6, f, N = 1)
        
    def state_vec_2_tuple(self, x):
        """
        Converts the state vector into the tuple (r, rDot, R, omega)
        Inputs:
            x (18x1 NumPy Array): state vector of system
        Reutns:
            r, rDot, R, omega: Tuple containing vectors/matrices
        """
        #assemble the state vector
        r = x[0: 3, 0].reshape((3, 1))
        rDot = x[3:6, 0].reshape((3, 1))
        R = x[6:15, 0].reshape((3, 3)).T
        omega = x[15:18, 0].reshape((3, 1))

        #return tuple
        return r, rDot, R, omega
    
    def input_vec_2_tuple(self, u):
        """
        Converts the input vector into the tuple (f, M)
        """
        f = u[0:3, :].reshape((3, 1))
        M = u[3:, :].reshape((3, 1))
        return f, M
    
    def rot_2_euler(self, Rdata):
        """
        Converts R data to a list of XYZ Euler Angles
        """
        #extract rotation matrices
        XEuler = []
        YEuler = []
        ZEuler = []
        for i in range(Rdata.shape[1]):
            #extract rotation matrix
            Ri = Rdata[:, i].reshape((3, 3)).T
            #convert to euler angles
            Rot = Rsp.from_matrix(Ri)
            eulerI = Rot.as_euler('XYZ')
            XEuler.append(eulerI[0])
            YEuler.append(eulerI[1])
            ZEuler.append(eulerI[2])
        return XEuler, YEuler, ZEuler
    
    def calc_rotation_error(self, R, Rd):
        return np.trace(self.I - Rd.T @ R)
    
    def calc_psi_hist(self, Rdata):
        #extract rotation matrices
        PsiHist = []
        for i in range(Rdata.shape[1]):
            #extract rotation matrix
            Ri = Rdata[:, i].reshape((3, 3)).T
            PsiHist.append(self.calc_rotation_error(Ri, self.Rd))
        return PsiHist
    
    def show_plots(self, xData, uData, tData, stateLabels=None, inputLabels=None, obsManager=None):
        """
        Plot the system behavior over time. Plots XY trajectory and Euler Angles.
        """
        #extract the states and times
        x0Hist = xData[0, :].tolist()
        y0Hist = xData[1, :].tolist()
        z0Hist = xData[2, :].tolist()
        tHist = tData[0, :]

        #extract rotation matrices
        Rhist = xData[6:15, :]
        XEuler, YEuler, ZEuler = self.rot_2_euler(Rhist)

        #plot the positions in space
        plt.plot(x0Hist, y0Hist)
        plt.title("Spatial Trajectory of Ball")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.show()

        #Plot the positions in time
        plt.plot(tHist, x0Hist)
        plt.plot(tHist, y0Hist)
        plt.xlabel("Time (s)")
        plt.ylabel("Position (m)")
        plt.legend(["X", "Y"])
        plt.title("Spatial Trajectory of Ball")
        plt.show()

        #Plot rotation error
        if self.Rd is not None:
            #compute rotation error across time
            psiHist = self.calc_psi_hist(Rhist)
            plt.plot(tHist, psiHist)
            plt.xlabel("Time (s)")
            plt.ylabel("Psi")
            plt.title("Evolution of Orientation Error")
            plt.show()

        #plot the euler angles
        showEuler = False
        if showEuler:
            plt.plot(tHist, XEuler)
            plt.plot(tHist, YEuler)
            plt.plot(tHist, ZEuler)
            plt.title("Evolution of XYZ Euler Angles")
            plt.legend(["X", "Y", "Z"])
            plt.xlabel("Time (s)")
        plt.show()

    def gen_sphere(self, R, r):
        """
        Generate mesh for sphere of radius rho at the origin with orientation R and position r
        """
        u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:30j]
        x = self.rho * np.cos(u) * np.sin(v)
        y = self.rho * np.sin(u) * np.sin(v)
        z = self.rho * np.cos(v)

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                rSphere = np.vstack((x[i, j], y[i, j], z[i, j]))
                #rotate r
                rSphere = R @ rSphere
                x[i, j], y[i, j], z[i, j] = rSphere[0, :] + r[0, 0], rSphere[1, :] + r[1, 0], rSphere[2, :] + self.rho
        return x, y, z

    def show_animation(self, xData, uData, tData, animate = True, obsManager = None):
        #Set constant animtion parameters
        FREQ = 50 #control frequency, same as data update frequency
        
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')

        x0Hist = xData[0, :].tolist()
        xMin, xMax = np.min(x0Hist), np.max(x0Hist)
        y0Hist = xData[1, :].tolist()
        yMin, yMax = np.min(y0Hist), np.max(y0Hist)
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
            r, rDot, R, omega = self.state_vec_2_tuple(state)

            #generate sphere
            x, y, z = self.gen_sphere(R, r)

            #plot the 3D surface on the axes
            ax.plot_surface(x, y, z, cmap=plt.cm.viridis, linewidth=0.1)

            return fig,
    
        num_frames = xData.shape[1]-1
        anim = animation.FuncAnimation(fig = fig, func = update, frames=num_frames, interval=1/FREQ*1000)
        plt.show()


class RollingBallNH(cs.Dynamics):
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
            rDot = self.rho * cs.hat(R @ omega) @ self.e3
            RDot = R @ cs.hat(omega)

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

    def rot_2_euler(self, Rdata):
        """
        Converts R data to a list of XYZ Euler Angles
        """
        return RollingBall.rot_2_euler(self, Rdata)
    
    def calc_rotation_error(self, R, Rd):
        return np.trace(self.I - Rd.T @ R)
    
    def calc_psi_hist(self, Rdata):
        #extract rotation matrices
        PsiHist = []
        for i in range(Rdata.shape[1]):
            #extract rotation matrix
            Ri = Rdata[:, i].reshape((3, 3)).T
            PsiHist.append(self.calc_rotation_error(Ri, self.Rd))
        return PsiHist
    
    def show_plots(self, xData, uData, tData, stateLabels=None, inputLabels=None, obsManager=None):
        """
        Plot the system behavior over time. Plots XY trajectory and Euler Angles.
        """
        #extract the states and times
        x0Hist = xData[0, :].tolist()
        y0Hist = xData[1, :].tolist()
        z0Hist = xData[2, :].tolist()
        tHist = tData[0, :]

        #extract rotation matrices
        Rhist = xData[3:, :]
        XEuler, YEuler, ZEuler = self.rot_2_euler(Rhist)

        #plot the positions versus time
        plt.plot(x0Hist, y0Hist)
        plt.title("Spatial Trajectory of Ball")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.show()

        #plot the euler angles
        plt.plot(tData[0, :], XEuler)
        plt.plot(tData[0, :], YEuler)
        plt.plot(tData[0, :], ZEuler)
        plt.title("Evolution of XYZ Euler Angles")
        plt.legend(["X", "Y", "Z"])
        plt.xlabel("Time (s)")
        plt.show()

        #plot the inputs
        omega1 = uData[0, :].tolist()
        omega2 = uData[1, :].tolist()
        omega3 = uData[2, :].tolist()
        plt.plot(tData[0, :], omega1)
        plt.plot(tData[0, :], omega2)
        plt.plot(tData[0, :], omega3)
        plt.title("Evolution of Angular Velocity Inputs")
        plt.legend(["wx", "wy", "wz"])
        plt.xlabel("Time (s)")
        plt.ylabel("Angular Velocity (Rad/s)")
        plt.show()

        #Plot rotation error
        if self.Rd is not None:
            #compute rotation error across time
            psiHist = self.calc_psi_hist(Rhist)
            plt.plot(tHist, psiHist)
            plt.xlabel("Time (s)")
            plt.ylabel("Psi")
            plt.title("Evolution of Orientation Error")
            plt.show()

        #plot the euler angles
        showEuler = False
        if showEuler:
            plt.plot(tHist, XEuler)
            plt.plot(tHist, YEuler)
            plt.plot(tHist, ZEuler)
            plt.title("Evolution of XYZ Euler Angles")
            plt.legend(["X", "Y", "Z"])
            plt.xlabel("Time (s)")
        plt.show()

    def gen_sphere(self, R, r):
        """
        Generate mesh for sphere of radius rho at the origin with orientation R and position r
        """
        return RollingBall.gen_sphere(self, R, r)

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
            r, R = self.state_vec_2_tuple(state)

            #generate sphere
            x, y, z = self.gen_sphere(R, r)

            #plot the 3D surface on the axes
            ax.plot_surface(x, y, z, cmap=plt.cm.viridis, linewidth=0.1)

            return fig,
    
        num_frames = xData.shape[1]-1
        anim = animation.FuncAnimation(fig = fig, func = update, frames=num_frames, interval=1/FREQ*1000)
        plt.show()

class RollingBallNHSimple(cs.Dynamics):
    """
    Rolling Ball Nonholonomic dynamics - Simple dynamics -> Right Invariant system
    transformed to have unit radius
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
        self.A = np.array([[0, 1, 0], [-1, 0, 0]])

        #Implement the state dynamics
        def f(x, u, t):
            """
            Dynamics function
            x = (r, R)
            u = (omega)
            """
            #extract components of state vector
            r, R = self.state_vec_2_tuple(x)

            #compute dynamics
            rDot = self.A @ u
            RDot = cs.hat(u) @ R

            #Reshape rotation matrix into a vector
            RDot = (RDot.T).reshape((9, 1))
            
            #stack the dynamics terms and return xDot
            xDot = np.vstack((rDot, RDot))
            return xDot
        
        #Call the super init function -> default to one ball
        super().__init__(x0, 11, 3, f, N = 1)
        
    def state_vec_2_tuple(self, x):
        """
        Converts the state vector into the tuple (r, rDot, R, omega)
        Inputs:
            x (11x1 NumPy Array): state vector of system
        Reutns:
            r, R: Tuple containing vectors/matrices
        """
        #assemble the state vector
        r = x[0: 2, 0].reshape((2, 1))
        R = x[2:, 0].reshape((3, 3)).T

        #return tuple
        return r, R

    def rot_2_euler(self, Rdata):
        """
        Converts R data to a list of XYZ Euler Angles
        """
        return RollingBall.rot_2_euler(self, Rdata)
    
    def calc_rotation_error(self, R, Rd):
        return np.trace(self.I - Rd.T @ R)
    
    def calc_psi_hist(self, Rdata):
        #extract rotation matrices
        PsiHist = []
        for i in range(Rdata.shape[1]):
            #extract rotation matrix
            Ri = Rdata[:, i].reshape((3, 3)).T
            PsiHist.append(self.calc_rotation_error(Ri, self.Rd))
        return PsiHist
    
    def show_plots(self, xData, uData, tData, stateLabels=None, inputLabels=None, obsManager=None):
        """
        Plot the system behavior over time. Plots XY trajectory and Euler Angles.
        """
        #extract the states and times
        x0Hist = xData[0, :].tolist()
        y0Hist = xData[1, :].tolist()
        tHist = tData[0, :]

        #extract rotation matrices
        Rhist = xData[2:, :]
        XEuler, YEuler, ZEuler = self.rot_2_euler(Rhist)

        #plot the positions versus time
        plt.plot(x0Hist, y0Hist)
        plt.title("Spatial Trajectory of Ball")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.show()

        #plot the euler angles
        plt.plot(tData[0, :], XEuler)
        plt.plot(tData[0, :], YEuler)
        plt.plot(tData[0, :], ZEuler)
        plt.title("Evolution of XYZ Euler Angles")
        plt.legend(["X", "Y", "Z"])
        plt.xlabel("Time (s)")
        plt.show()

        #plot the inputs
        omega1 = uData[0, :].tolist()
        omega2 = uData[1, :].tolist()
        omega3 = uData[2, :].tolist()
        plt.plot(tData[0, :], omega1)
        plt.plot(tData[0, :], omega2)
        plt.plot(tData[0, :], omega3)
        plt.title("Evolution of Angular Velocity Inputs")
        plt.legend(["wx", "wy", "wz"])
        plt.xlabel("Time (s)")
        plt.ylabel("Angular Velocity (Rad/s)")
        plt.show()

        #Plot rotation error
        if self.Rd is not None:
            #compute rotation error across time
            psiHist = self.calc_psi_hist(Rhist)
            plt.plot(tHist, psiHist)
            plt.xlabel("Time (s)")
            plt.ylabel("Psi")
            plt.title("Evolution of Orientation Error")
            plt.show()

        #plot the euler angles
        showEuler = False
        if showEuler:
            plt.plot(tHist, XEuler)
            plt.plot(tHist, YEuler)
            plt.plot(tHist, ZEuler)
            plt.title("Evolution of XYZ Euler Angles")
            plt.legend(["X", "Y", "Z"])
            plt.xlabel("Time (s)")
        plt.show()

    def gen_sphere(self, R, r):
        """
        Generate mesh for sphere of radius rho at the origin with orientation R and position r
        """
        return RollingBall.gen_sphere(self, R, r)

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
            r, R = self.state_vec_2_tuple(state)

            #generate sphere -> append a zero to the position for z
            x, y, z = self.gen_sphere(R, np.vstack((r, 0)))

            #plot the 3D surface on the axes
            ax.plot_surface(x, y, z, cmap=plt.cm.viridis, linewidth=0.1)

            return fig,
    
        num_frames = xData.shape[1]-1
        anim = animation.FuncAnimation(fig = fig, func = update, frames=num_frames, interval=1/FREQ*1000)
        plt.show()

class RollingBallNHQuat(cs.Dynamics):
    """
    Rolling Ball Nonholonomic dynamics - Simple dynamics -> Quaternion, unit radius ball.
    """
    def __init__(self, x0):
        """
        Init function for rolling ball dynamics objet
        Inputs:
            x0 (6x1 NumPy Array): (x, q) initial condition
            Note: q is a quaternion of the form (q0, qVec)
        """
        #Define a desired rotation/position attribute (used for plotting in control)
        self.qd = None
        self.rd = None

        #Define e3 basis vector and its hat map
        self.I = np.eye(3)

        #define ball radius
        self.rho = 1

        #define A matrix as a funcntion of the quaternion
        Atop = np.array([[0, 1, 0], [-1, 0, 0]]) #position dynamics A
        self.A = lambda q: np.vstack((Atop, 0.5* q[1:].T, 0.5*(q[0, 0] * self.I + cs.hat(q[1:]))))

        #Implement the state dynamics
        def f(x, u, t):
            """
            Dynamics function
            x = (x, q)
            u = (omega)
            """
            #extract components of state vector
            q = x[2:]
            x = x[0:2]

            #stack the dynamics terms and return xDot
            xDot = self.A(q) @ u
            return xDot
        
        #Call the super init function -> default to one ball
        super().__init__(x0, 6, 3, f, N = 1)


    def show_plots(self, xData, uData, tData, stateLabels=None, inputLabels=None, obsManager=None):
        """
        Plot the system behavior over time. Plots XY trajectory and Euler Angles.
        """
        super().show_plots(xData, uData, tData)

        #extract the states and times
        x0Hist = xData[0, :].tolist()
        y0Hist = xData[1, :].tolist()
        tHist = tData[0, :]

        #plot the positions versus time
        plt.plot(x0Hist, y0Hist)
        plt.title("Spatial Trajectory of Ball")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.show()

    def gen_sphere(self, R, r):
        """
        Generate mesh for sphere of radius rho at the origin with orientation R and position r
        """
        return RollingBall.gen_sphere(self, R, r)

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
            q = state[2:]
            x = state[0:2]

            #convert q to a rotation
            qSP = np.vstack((q[1:], q[0])).reshape(4, )
            R = Rsp.from_quat(qSP).as_matrix()

            #generate sphere -> append a zero to the position for z
            x, y, z = self.gen_sphere(R, np.vstack((x, 0)))

            #plot the 3D surface on the axes
            ax.plot_surface(x, y, z, cmap=plt.cm.viridis, linewidth=0.1)

            return fig,
    
        num_frames = xData.shape[1]-1
        anim = animation.FuncAnimation(fig = fig, func = update, frames=num_frames, interval=1/FREQ*1000)
        plt.show()