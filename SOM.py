import numpy as np
import matplotlib.pyplot as plt

class Frame(object):
    def __init__(self,frame_dimension=2,frame_size=[1280,1080],num_p=10):
        '''
        : frame_dimension  The frame is 2D or 3D
        : frame_size       The size of the canvas of frame e.g. [1280, 1080]
        : num_particle     The quantity of particles
        '''
        self.frame_dimension = frame_dimension
        self.frame_size = frame_size
        self.num_particles = num_p
        self.particles = []

    def random_particle(self):
        self.particles = np.zeros((self.num_particles,self.frame_dimension))
        
        for i in range(self.frame_dimension):
            self.particles[:,i] = np.random.randint(0,self.frame_size[i],size = self.num_particles)
        print(self.particles)

    def pass_particles(self):
        pass
    
class SOM(object):
    def __init__(self,layer_num,layer_size,layer_dimension=2):
        '''
        : layer_num        The quantity of sub-network. (typically two)
        : layer_dimension  The dimension of neuron (2D, 3D or more)
        : layer_size       Should be a list of how many neurons in each subnetwork e.g. [n1,n2]
                           The length should be the same with layer_num.
        '''
        self.layer = []
        for i in range(layer_num):
            self.layer.append(np.zeros((layer_size[i],layer_dimension)))

som = SOM
f = Frame(2,[1000,1000],100)
f.random_particle()
