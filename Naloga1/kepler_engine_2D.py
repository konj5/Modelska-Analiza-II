import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm

from scipy.integrate import solve_ivp

def solve(method, y0, tmax, dtmax):
    sol = solve_ivp(f, [0,tmax], y0, method, max_step = dtmax, dense_output=True)
    return sol


def solve_euler(y0, tmax, dtmax):
    dt = dtmax
    ts = np.arange(0,tmax,dt)
    ys = [np.array(y0)]
    for i in range(len(ts)):
        ys.append(ys[-1] + dt * f(ts[i], ys[-1]))

    return ts, ys


def solve_wrapped(method, y0, tmax, dtmax):
    if method == "euler": return solve_euler(y0,tmax, dtmax)
    return solve(method, y0, tmax, dtmax)

class Object:
    def __init__(self, x0, y0, vx0, vy0, m):
        self.x0 = x0
        self.y0 = y0
        self.vx0 = vx0
        self.vy0 = vy0
        self.m = m

class SolarSystem:
    def __init__(self):
        self.objects = []
        self.connections = []

    def add_object(self, object):
        self.objects.append(object)

    def add_connection(self, target, cause):
        self.connections.append((target, cause))

    def setup_f(self):
        self.statevector = []
        
        for object in self.objects:
            self.statevector.append(object.x0)
            self.statevector.append(object.y0)
            self.statevector.append(object.vx0)
            self.statevector.append(object.vy0)

        def f(t,y):
            ydot = np.zeros_like(y)
            for connection in self.connections:
                i,j = connection
                ydot[4*i+0] = y[4*i+2]
                ydot[4*i+1] = y[4*i+3]

                ydot[4*i+2] = -self.objects[j].m*(y[4*i+0]-y[4*j+0]) / ((y[4*i+0]-y[4*j+0])**2+ (y[4*i+1]-y[4*j+1])**2)**(3/2)
                ydot[4*i+3] = -self.objects[j].m*(y[4*i+1]-y[4*j+1]) / ((y[4*i+0]-y[4*j+0])**2+ (y[4*i+1]-y[4*j+1])**2)**(3/2)

            return ydot
        return f
    
    def evolve(self, method, y0, tmax, dtmax):
        return solve_wrapped(method, y0, tmax, dtmax)
    
"""SolS = SolarSystem()
SolS.add_object(Object(0,0,0,0,1))
SolS.add_object(Object(1,0,0,0.5,1))
SolS.add_connection(1,0)
f = SolS.setup_f()
solution = SolS.evolve("RK23", SolS.statevector, tmax=10, dtmax=0.1)

ts = np.linspace(0,10, 1000)
ys = solution.sol(ts)


plt.plot(ys[4*1+0,:], ys[4*1+1,:])
plt.gca().set_aspect('equal')
plt.show()
"""

"""SolS = SolarSystem()
#SolS.add_object(Object(-0.5,0,0,0.5,1))
#SolS.add_object(Object(0.5,0,0,-0.5,1))
#SolS.add_object(Object(2,0,0,1,1))

SolS.add_object(Object(-0.2,0,0,0.5,0.5))
SolS.add_object(Object(0.2,0,0,-0.5,0.5))
SolS.add_object(Object(1,0,0,1,1))

SolS.add_connection(1,0)
SolS.add_connection(0,1)

SolS.add_connection(2,0)
SolS.add_connection(2,1)

f = SolS.setup_f()
tmax = 1000
solution = SolS.evolve("RK23", SolS.statevector, tmax=tmax, dtmax=0.1)

ts = np.linspace(0,tmax, 10*tmax)
ys = solution.sol(ts)


plt.plot(ys[4*0+0,:], ys[4*0+1,:])
plt.plot(ys[4*1+0,:], ys[4*1+1,:])
plt.plot(ys[4*2+0,:], ys[4*2+1,:])
plt.gca().set_aspect('equal')
plt.show()"""
