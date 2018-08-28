import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from nearest_neighbor import nb
from mpl_toolkits.mplot3d import Axes3D
from cov import cov
from Frame import Frame
##from dynamic_plot import animate
import os

def file_is_empty(fpath):
    try:
        if os.path.getsize(fpath) > 0:
            return False
        else:
            return True
    except OSError as e:
        return True

class SOM(object):
    def __init__(self,layer_num = 2,layer_size = [10,10],layer_dimension=2):
        '''
        : layer_num        The quantity of sub-network. (typically two)
        : layer_dimension  The dimension of neuron (2D, 3D or more)
        : layer_size       Should be a list of how many neurons in each subnetwork e.g. [n1,n2]
                           The length should be the same with layer_num.
        '''
        self.layer_num = layer_num
        self.layer = []
        for i in range(layer_num):
            self.layer.append(np.zeros((layer_size[i],layer_dimension)))

    def generateGaussian(self,start_value,end_value,sigma,howMany = 10):
        '''
        : start _value     This is the value when i = 0
        : end_value        
        : sigma            This is how the value changed, the smaller the sigma, more severe the value is changed
        : howMany          How many values to generate
        '''
        x0 = 0
        x1 = np.sqrt(-sigma**2 * np.log(end_value / start_value))

        x = np.linspace(x0,x1,num = howMany, endpoint = True)
        ret = []
        for i in range(howMany):
            ret.append(start_value * np.exp(-x[i]**2/sigma**2))
        return np.array(ret)

    def findClosest(self,point,layer):
        '''
        : point            The stimuli particle
        : layer            The particles that will be examined the smallest distance with point
        : return           The closest particle (winner) in layer and the distance between point and the winner
        '''
        if len(point) != layer.shape[1]:
            print('point and layer particle dimension does not match')
            return

        dist = np.linalg.norm(point-layer,axis = 1)

        minIndex = np.argmin(dist)
        winner = layer[minIndex]

        distance = point - winner # not 'np.linalg.norm(point - winner)' 
        return winner, distance

    def move(self,layer,winner,displacement,sigma,alpha):
        '''
        : layer            The particles to move
        : winner           The center to determine the displacement
        : displacement     The distance to move particles
        : sigma            The sigma to divide e.g. exp(-d**2/sigma**2)
        '''

        for i in range(len(layer)):
            dist = np.linalg.norm(layer[i]-winner)
            #print('dist',dist)
            layer[i] += alpha * displacement * np.exp(-dist**2 / sigma**2)

    def move_direction(self,layer,winner,displacement,sigma,alpha,direction):
        '''
        : layer            The particles to move
        : winner           The center to determine the displacement
        : displacement     The distance to move particles
        : sigma            The sigma to divide e.g. exp(-d**2/sigma**2)
        : direction        The direction to determine the angle between direction and displacement
        '''
        for i in range(len(layer)):
            dist = np.linalg.norm(layer[i]-winner)

            #print('angle is',self.find_angle(displacement,v1 = direction))
            if self.find_angle(displacement,v1 = direction) < 30:
                alpha *= 0

            
            layer[i] += alpha * displacement * np.exp(-dist**2 / sigma**2)
            
    def checkSimilarity(self,layer0,layer1):
        '''
        : layer0, layer1   The two sets of particles to check similarity
        : return           The similarity between two sets of particles, e.g. 100%
        '''

        threshold = 1
        count = 0
        pair = []
        '''
        : choose the layer with smaller length as the one to check similarity percent
        '''

        if len(layer0) < len(layer1):
            for i in range(len(layer0)):
                for j in range(len(layer1)):
                    if np.linalg.norm(layer0[i]-layer1[j]) <= threshold:
                        count += 1
                        pair.append([i,j])
                        break
            return count / layer0.shape[0]
        else:
            for i in range(len(layer1)):
                for j in range(len(layer0)):
                    if np.linalg.norm(layer1[i]-layer0[j]) <= threshold:
                        count += 1
                        pair.append([j,i])
                        break
            return count / layer1.shape[0], pair         
                
    def draw2Frame(self,layer0,layer1,pair,_3d = False):
        '''
        : layer0 & layer1  The sets of particles
        : pair             This is a list of paired particles [a,b] a is the index of the first layer and b is the index of the second layer
        '''
        if _3d == False:
            for i in pair:
                plt.arrow(layer0[i[0]][0],layer0[i[0]][1],
                          layer1[i[1]][0]-layer0[i[0]][0],layer1[i[1]][1]-layer0[i[0]][1],)
                          #length_includes_head=True)
            plt.scatter(layer0[:,0],layer0[:,1],color = 'r',s=50,marker = 'o',label='layer0')
            plt.scatter(layer1[:,0],layer1[:,1],color = 'g',s=50,marker = 'x',label='layer1')
            
            #plt.show()
        else:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.scatter(layer0[:,0],layer0[:,1],layer0[:,2],c='r',s=50,marker='o',label='layer0')
            ax.scatter(layer1[:,0],layer1[:,1],layer1[:,2],c='g',s=50,marker='x',label='layer1')

            for i in pair:
                u = layer1[i,0] - layer0[i,0]
                v = layer1[i,1] - layer0[i,1]
                w = layer1[i,2] - layer0[i,2]

                ax.quiver(layer0[i,0],layer0[i,1],layer0[i,2],u,v,w,pivot='tail')

        #plt.show()  #eddited recently

    def find_angle(self,v2,v1 = np.array([1,1,1]),original = np.array([0,0,0])):
        '''
        : v2, v1           The second vector and the first vector
        : original         The zero point to normalize two vectors
        '''
        v2 = np.array(v2)
        v1 = np.array(v1)
            
        v2 = v2 - original
        v1 = v1 - original

        cosine_angle = np.dot(v2,v1) / (np.linalg.norm(v1) * np.linalg.norm(v2))

##        print('dot product is ',(np.dot(v2,v1)))
##        print('v1 norm is',np.linalg.norm(v1))
##        print('v2 norm is',np.linalg.norm(v2))
##        print('cos angle is ',cosine_angle)
        if np.absolute(cosine_angle  - 1) < 0.0001:
            return 0
        else:
            return np.arccos(cosine_angle) * 180 / np.pi
                
    def fit_2d_2frames(self,frame1,frame2,epoch):
        '''
        : frame 1          The first layer's initiate value
        : frame 2          The second layer's initiate value
        : epoch            The iteration times in fit, one epoch means update the second layer by the first and then update the first layer by the second
        '''
        #print('self layer num',self.layer_num)



        if self.layer_num != 2:
            print('layer number does not match')
        else:
            self.layer[0] = frame1.particles
            self.layer[1] = frame2.particles

        layer0 = np.array(self.layer[0]).copy()
        layer1 = np.array(self.layer[1]).copy()
        #layer1 = np.array(self.layer[0]).copy()

        '''
        : alpha is the initial coefficient to move particle  movement = alpha * distance
        : r     is the initial distance between particles
        '''

        alpha = 0.1
        r0 = frame1.statistics_2d()
        r1 = frame2.statistics_2d()

        print('similarity before',self.checkSimilarity(layer0,layer1)[0])

        a = self.generateGaussian(alpha,0.01,1,epoch)

        r00 = self.generateGaussian(r0,0.3*r0,1,epoch)
        r11 = self.generateGaussian(r1,0.3*r0,1,epoch)
        #print(layer1)
        #print('   ')

        if file_is_empty('./accuracy_normal.txt') == False:
            os.remove('./accuracy_normal.txt')
                
        for e in range(epoch):
            for i in range(len(layer0)):
                winner, displacement = self.findClosest(layer0[i],layer1)
                self.move(layer1,winner,displacement,r11[e],a[e])
            print('in ' + str(e) + ' epoch' + ' the accuracy is: ')
            accu,_ = self.checkSimilarity(layer0,layer1)
            print(accu)


            with open('accuracy_normal.txt', 'a') as the_file:
                the_file.write(str(e)+ ',' + str(accu) + '\n')
                
        #print(layer1)
        _ , pair = self.checkSimilarity(layer0,layer1)
        print('similarity after',self.checkSimilarity(layer0,layer1)[0])

        
        if frame1.frame_dimension == 2:
            self.draw2Frame(frame1.particles,frame2.particles,pair,_3d = False)
            plt.scatter(frame1.frame_size[0]/2,frame1.frame_size[1]/2,color = 'b',s=50,marker = 'v',label='center') # plot the center of frame
        elif frame1.frame_dimension == 3:
            self.draw2Frame(frame1.particles,frame2.particles,pair,_3d = True)
        #print('difference',layer0-layer1)
        return self.checkSimilarity(layer0,layer1)[0] ## new added

    def fit_iteration(self,frame1,frame2,epoch):
        '''
        : frame 1          The first layer's initiate value
        : frame 2          The second layer's initiate value
        : epoch            The iteration times in fit, one epoch means update the second layer by the first and then update the first layer by the second
        '''
        #print('self layer num',self.layer_num)

        if self.layer_num != 2:
            print('layer number does not match')
        else:
            self.layer[0] = frame1.particles
            self.layer[1] = frame2.particles

        layer0 = np.array(self.layer[0]).copy()
        layer1 = np.array(self.layer[1]).copy()

        '''
        : alpha is the initial coefficient to move particle  movement = alpha * distance
        : r     is the initial distance between particles
        '''

        alpha = 0.1
        r0 = frame1.statistics_2d()
        r1 = frame2.statistics_2d()

        print('similarity before',self.checkSimilarity(layer0,layer1)[0])

        a = self.generateGaussian(alpha,0.01,1,epoch)

        r00 = self.generateGaussian(r0,0.3*r0,1,epoch)
        r11 = self.generateGaussian(r1,0.3*r0,1,epoch)

        if file_is_empty('./accuracy_iteration.txt') == False:
            os.remove('./accuracy_iteration.txt')

        for e in range(epoch):
            for i in range(len(layer0)):
                winner, displacement = self.findClosest(layer0[i],layer1)

                direction = [1,1,1]
                
                self.move_direction(layer1,winner,displacement,r11[e],a[e],direction)
            print('in ' + str(e) + ' epoch' + ' the accuracy is: ')
            accu,_ = self.checkSimilarity(layer0,layer1)
            print(accu)

            with open('accuracy_iteration.txt', 'a') as the_file:
                the_file.write(str(e)+ ',' + str(accu) + '\n')

        #print(layer1)
        _ , pair = self.checkSimilarity(layer0,layer1)
        print('similarity after',self.checkSimilarity(layer0,layer1)[0])

        
        if frame1.frame_dimension == 2:
            self.draw2Frame(frame1.particles,frame2.particles,pair,_3d = False)
            plt.scatter(frame1.frame_size[0]/2,frame1.frame_size[1]/2,color = 'b',s=50,marker = 'v',label='center') # plot the center of frame
        elif frame1.frame_dimension == 3:
            self.draw2Frame(frame1.particles,frame2.particles,pair,_3d = True)
        #print('difference',layer0-layer1)
        return self.checkSimilarity(layer0,layer1)[0] ## new added