import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm

from scipy.integrate import solve_ivp


def f(t,y):
    ydot = np.zeros((4))

    ydot[0] = y[2]
    ydot[1] = y[3]
    ydot[2] = -y[0] / (y[0]**2+ y[1]**2)**(3/2)
    ydot[3] = -y[1] / (y[0]**2+ y[1]**2)**(3/2)

    return ydot

def H(y):
    return (y[2]**2 + y[3]**2)/2 - 1/np.sqrt(y[0]**2 + y[1]**2)

def L(y):
    return y[0] * y[3] - y[1] * y[2]

def A(y):
    return L(y) * np.array([y[3], -y[2]]) - np.array([y[0], -y[1]])/np.sqrt(y[0]**2 + y[1]**2)


def solve(method, y0, tmax):
    sol = solve_ivp(f, [0,tmax], y0, method, max_step = 0.1, dense_output=True)
    return sol


def solve_euler(y0, tmax):
    dt = 0.05
    ts = np.arange(0,tmax,dt)
    ys = [np.array(y0)]
    for i in range(len(ts)):
        ys.append(ys[-1] + dt * f(ts[i], ys[-1]))

    return ts, ys


tmax = 10
sol = solve("RK45", [1,0,0,1], tmax)

#ts, ys = sol.t, sol.y

ts = np.linspace(0,tmax, 100)
ys = sol.sol(ts)


plt.plot(ys[0,:], ys[1,:])
plt.gca().set_aspect('equal')
plt.show()