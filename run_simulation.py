import numpy as np
import matplotlib.pyplot as plt
from dynamics_simulator import *
import CalSim as cs
from scipy.spatial.transform import Rotation as R
from controllers import *

#set an initial condition -> NOTE: Must transpose R0 before sending in!
r0 = np.zeros((3, 1))
R0 = (cs.calc_Rx(np.pi/2) @ cs.calc_Ry(np.pi/2)).T.reshape((9, 1)) #np.eye(3).reshape((9, 1))
omega0 = np.zeros((3, 1))

#define dynamics object
ballDyn = RollingBall(r0, R0, omega0)

#create an observer
observerManager = cs.ObserverManager(ballDyn)

#create a snake controller manager
controllerManager = cs.ControllerManager(observerManager, BallCLF)

#create a ball environment
env = cs.Environment(ballDyn, controllerManager, observerManager, T = 30)

#run the simulation
env.run()
