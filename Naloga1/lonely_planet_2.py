import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import seaborn as sns

from scipy.integrate import solve_ivp
import kepler_engine_2D as kepler

def Hi(system,i,y):
    v = np.sqrt(y[4*i+2]**2 + y[4*i+3]**2)
    H = 1/2 * system.objects[i].m * v
    for connection in system.connections:
        a,b = connection
        if a != i: continue

        rab = np.sqrt((y[4*a+0]-y[4*b+0])**2 + (y[4*a+1]-y[4*b+1])**2)

        H += -system.objects[a].m*system.objects[b].m / rab

    return H

def H(system,y):
    H = 0
    for i in range(len(system.objects)):
        H += Hi(system, i, y)
    return H

def Li(system,i,y):
    return system.objects[i].m * np.cross([y[4*i+0], y[4*i+1], 0], [y[4*i+2], y[4*i+3], 0])[2]

def L(system,y):
    L = 0
    for i in range(len(system.objects)):
        L += Li(system, i, y)
    return L

def Ai(system, y, i):
    Li = system.objects[i].m * np.cross([y[4*i+0], y[4*i+1], 0], [y[4*i+2], y[4*i+3], 0])

    A = np.cross([y[4*i+2], y[4*i+3], 0], Li)

    for connection in system.connections:
        a,b = connection
        if a != i: continue

        A += -system.objects[a].m*system.objects[b].m *np.array([y[4*a+0]-y[4*b+0], y[4*a+1]-y[4*b+1], 0])/np.linalg.norm(np.array([y[4*a+0]-y[4*b+0], y[4*a+1]-y[4*b+1], 0]))

    return A

##### Various orbits
tmax = 100
v0s = np.linspace(0,3,10)
ts = np.linspace(0,tmax, 10*tmax)
yss = np.zeros((len(v0s), len(ts), 4*2))
for i, v0 in enumerate(v0s):
    system = kepler.SolarSystem()
    system.add_object(kepler.Object(0,0,0,0,1))
    system.add_object(kepler.Object(1,0,0,v0,1))
    system.add_connection(1,0)

    system.setup_f()

    solution = system.evolve("RK45", system.statevector, tmax, dtmax=0.1)
    yss[i,:,:] = solution.sol(ts)

cmap = plt.get_cmap("viridis")
norm = colors.Normalize(v0[0], v0[-1])

for i in range(len(v0s)):
    plt.plot(yss[i, :, 0], yss[i, :, 1], c = cmap(norm(v0s[i])))

plt.colorbar(cm.ScalarMappable(norm,cmap))

plt.show()







