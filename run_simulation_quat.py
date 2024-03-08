import numpy as np
import matplotlib.pyplot as plt
from dynamics_simulator import *
import CalSim as cs
from scipy.spatial.transform import Rotation
from quat_nh_controllers import *

#NONHOLONOMIC KINEMATICS SIMULATION

#set an initial condition -> NOTE: Must transpose R0 before sending in!
r0 = 0 * np.ones((2, 1))
R0 = np.eye(3)
q0SP = Rotation.from_matrix(R0).as_quat() #scipy quaternion
q0 = np.hstack([q0SP[3], q0SP[0:3]]).reshape((4, 1))
x0 = np.vstack((r0, q0))

#define dynamics object
ballDyn = RollingBallNHQuat(x0)

#create an observer
observerManager = cs.ObserverManager(ballDyn)

#create a snake controller manager
controllerManager = cs.ControllerManager(observerManager, MPCBallQuat)

#create a ball environment
env = cs.Environment(ballDyn, controllerManager, observerManager, T = 4)

#run the simulation
env.run()
