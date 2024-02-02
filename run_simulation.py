import numpy as np
import matplotlib.pyplot as plt
from dynamics_simulator import *
import CalSim as cs
from scipy.spatial.transform import Rotation as R


#set an initial condition
r0 = 0 * np.ones((3, 1))
R0 = np.eye(3).reshape((9, 1))
omega0 = 5 * np.array([[0.75, 2, 1]]).T

#define dynamics object
ballDyn = RollingBall(r0, R0, omega0)

#create an observer
observerManager = cs.ObserverManager(ballDyn)

#create a snake controller manager
controllerManager = None

#create a snake environment
env = cs.Environment(ballDyn, controllerManager, observerManager, T = 2)

#run the simulation
env.run()
