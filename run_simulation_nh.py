import numpy as np
import matplotlib.pyplot as plt
from dynamics_simulator import *
import CalSim as cs
from scipy.spatial.transform import Rotation as R
from controller import *

#NONHOLONOMIC KINEMATICS SIMULATION

#set an initial condition -> NOTE: Must transpose R0 before sending in!
r0 = 0 * np.ones((3, 1))
R0 = (cs.calc_Rx(np.pi/2) @ cs.calc_Ry(np.pi/2)).T.reshape((9, 1)) #np.eye(3).reshape((9, 1))
x0 = np.vstack((r0, R0))

#define dynamics object
ballDyn = RollingBallNH(x0)

#create an observer
observerManager = cs.ObserverManager(ballDyn)

#create a snake controller manager
controllerManager = cs.ControllerManager(observerManager, EulerPlanner)

#create a snake environment
env = cs.Environment(ballDyn, controllerManager, observerManager, T = 10)

#run the simulation
env.run()
