import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as Rsp


x = np.arange(0,10,0.1)
y = np.arange(0,10,0.1)
X,Y = np.meshgrid(x,y)


fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')


def animate(n):
    ax.cla()

    # u = np.sin((X + Y) + n)

    # ax.plot_surface(X, Y, u)
    ax.set_zlim(-2, 2)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    # get a grid of points

    u, v = np.mgrid[0:2 * np.pi:15j, 0:np.pi:15j]
    x = 0.5 * np.cos(u) * np.sin(v)
    y = 0.5 * np.sin(u) * np.sin(v)
    z = 0.5 * np.cos(v)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            r = np.vstack((x[i, j], y[i, j], z[i, j]))

            #rotate about z axis
            Rzn = Rsp.from_euler("ZYX", [n/10, 0, 0]).as_matrix()
            r = Rzn @ r
            x[i, j], y[i, j], z[i, j] = r[0, :], r[1, :], r[2, :]

    #plot the 3D surface on the axes
    ax.plot_surface(x, y, z, cmap=plt.cm.viridis, linewidth=0.1)

    return fig,


anim = FuncAnimation(fig = fig, func = animate, frames = 100, interval = 1, repeat = True)
plt.show()