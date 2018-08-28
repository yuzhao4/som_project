import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from nearest_neighbor import nb
from mpl_toolkits.mplot3d import Axes3D
from cov import cov
from Frame import Frame
##from dynamic_plot import animate
import os
from SOM_CLASS import SOM

np.random.seed(19680801)

PARTICLE_NUM = 30
ITERATION = 100

PHI_LEFT = 1.5
PHI_RIGHT = 1
NUM_PHI = 1

PHI = np.linspace(PHI_LEFT,PHI_RIGHT,NUM_PHI)

f1 = Frame(3,[1000,1000,1000],PARTICLE_NUM)

distance = f1.statistics_2d()

f1.random_particle()

velocities = []

for p in PHI:
    v = distance / p
    print('v',v)
    velocities.append([v,v,v])
    
def main1():
    som = SOM()

    results = []

    for i in range(len(PHI)):
        f2 = f1.move_particles(velocities[i])

        som_accu = som.fit_2d_2frames(f1,f2,ITERATION)

        _,nb_accu = nb(f1,f2,velocities[i])

        results.append([som_accu,nb_accu,PHI[i]])

    with open('results_raw.txt', 'w') as filehandle:  
        for listitem in results:
            filehandle.write('%s\n' % listitem)    

def main_iteration():
    som = SOM()

    results = []

    for i in range(len(PHI)):
        f2 = f1.move_particles(velocities[i])

        som_accu = som.fit_iteration(f1,f2,ITERATION)

        _,nb_accu = nb(f1,f2,velocities[i])

        results.append([som_accu,nb_accu,PHI[i]])

    with open('results_iteration.txt', 'w') as filehandle:  
        for listitem in results:
            filehandle.write('%s\n' % listitem)   

import threading
threads = []

t1 = threading.Thread(target=main1)
t2 = threading.Thread(target=main_iteration)

t1.start()
t2.start()

