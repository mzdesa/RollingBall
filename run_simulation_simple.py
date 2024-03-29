import numpy as np
import matplotlib.pyplot as plt
from dynamics_simulator import *
import CalSim as cs
from scipy.spatial.transform import Rotation as R
from simple_nh_controllers import *

#NONHOLONOMIC KINEMATICS SIMULATION

#set an initial condition -> NOTE: Must transpose R0 before sending in!
r0 = 0 * np.ones((2, 1))
R0 = np.eye(3).reshape((9, 1))
x0 = np.vstack((r0, R0))

#define dynamics object
ballDyn = RollingBallNHSimple(x0)

#create an observer
observerManager = cs.ObserverManager(ballDyn)

#create a snake controller manager
controllerManager = cs.ControllerManager(observerManager, MPCBall)

#create a ball environment
env = cs.Environment(ballDyn, controllerManager, observerManager, T = 10)

#run the simulation
env.run()
