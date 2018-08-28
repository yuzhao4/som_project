import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from nearest_neighbor import nb
from mpl_toolkits.mplot3d import Axes3D
from cov import cov

##from dynamic_plot import animate
import os

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
        #print(self.particles)

    def move_rotation_2d(self,degree,center=True,clockwise=True):
        '''
        : degree           The degree to rotate clockwise or non-clockwis
        : center           The center to rotate
        : clockwise        The direction to rotate
        '''
        temp_frame = self.copy()
        
        center_x = temp_frame.frame_size[0]/2
        center_y = temp_frame.frame_size[1]/2

        def normalize(particle,center_x,center_y):
            x = particle[0]
            y = particle[1]
            
            x = x - center_x
            y = y - center_y
            #print('x',x,'y',y)
            r = np.sqrt(x**2 + y**2)
            #print('r',r)
            if x >= 0 and y >= 0:
                theta = np.arcsin(y / r)
                theta = theta * 180 / np.pi
            elif x <0 and y >= 0:
                theta = np.arcsin(y / r)
                theta = (np.pi - theta) * 180 / np.pi
            elif x < 0 and y < 0:
                theta = np.arcsin(- y / r)
                theta = (theta + np.pi) * 180 / np.pi
            else:
                theta = np.arcsin(y / r)
                theta = theta * 180 / np.pi   
            return r, theta
            
        if center == True:
            if clockwise == True:
                for particle in temp_frame.particles:
                    r, theta = normalize(particle,center_x,center_y)
                    theta -= degree

                    particle[0] = r * np.cos(theta * np.pi / 180) + center_x
                    particle[1] = r * np.sin(theta * np.pi / 180) + center_y
                return temp_frame
            else:
                for particle in self.particles:
                    r, theta = normalize(particle,center_x,center_y)
                    theta += degree

                    particle[0] = r * np.cos(theta * np.pi / 180) + center_x
                    particle[1] = r * np.sin(theta * np.pi / 180) + center_y
                return temp_frame
        else:
            print('to do, non-center rotation')

    # This function is to generate a frame that has same size and same particle number that has a small displacement to test the SOM fit method
    def move_particles(self,velocity):
        '''
        : velocity         The velocity vector, should be the same dimension with frame_dimension
        '''
        if len(velocity) == self.frame_dimension:
            temp_frame = self.copy()
            #print('before particles',temp_frame.particles)
            temp_frame.particles += velocity
            #print('moved successfully')
            #print('after particles',temp_frame.particles)
            return temp_frame
        else:
            print("the velocity vecotr dimension doesn't match frame_dimension")

    def copy(self):
        temp_frame = Frame()
        temp_frame.frame_dimension = self.frame_dimension
        temp_frame.frame_size = self.frame_size.copy()
        temp_frame.num_particles = self.num_particles
        temp_frame.particles = self.particles.copy() # use np.array.copy() to avoid shallow copy
        return temp_frame

    def statistics_2d(self):
        '''
        : self            The frame to analysis the average distance of particles in this frame
        : return           The average distance
        '''
        dimension = self.frame_dimension
        
        volume = 1
        for axis in self.frame_size:
            volume *= axis
        density = np.sqrt(self.num_particles / volume)

        if dimension == 2:        
            distance = np.sqrt(volume / (np.pi * self.num_particles))
            return distance
        elif dimension == 3:
            distance = (3 * volume / (4 * np.pi * self.num_particles))** (1/3)
            return distance
##        distance = []
        
##        for i in range(dimension):
##            left = np.argmin(self.particles[:,i])
##            right = np.argmax(self.particles[:,i])
##
##            dist = self.particles[right][i] - self.particles[left][i]
##            distance.append(dist)
##     
##        distance = np.array(distance) ** 2
##        return np.sqrt(np.sum(distance))/((self.num_particles)**(1/3)*2)        
