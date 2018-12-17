import numpy as np
import matplotlib.pyplot as plt

def nb(frame1,frame2,velocity):
    '''
    : frame1 and frame2      The two frame class, the second class is searched based on frame1
    '''
    
    p1 = frame1.particles.copy()
    p2 = frame2.particles.copy()

    pairs = []
    count = 0

    for i in range(p1.shape[0]):
        dist = np.linalg.norm(p2-p1[i],axis = 1)
        #print('dist',dist)
        minIndex = np.argmin(dist)
        if minIndex == i:
            count += 1
        #closest_particle = p2[minIndex]

        #pairs.append([pp1,closest_particle])
    
##    for pp1 in p1:
##        dist = np.linalg.norm(p2-pp1,axis = 1)
##        #print('dist',dist)
##        minIndex = np.argmin(dist)
##
##        closest_particle = p2[minIndex]
##
##        pairs.append([pp1,closest_particle])
    
##    accu = 0
##    for i in pairs:
##        if i[0][0] + velocity[0] == i[1][0] and i[0][1] + velocity[1] == i[1][1]:
##            accu += 1
##    #print('pairs',pairs)
##    accu = accu / len(pairs)
    
    return pairs,count / p1.shape[0]
