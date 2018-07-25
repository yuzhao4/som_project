# this file is used to generate particles at one frame
import numpy as np
import matplotlib.pyplot as plt

global width
global height

width = 1280
height = 1080

# how to make a constructor??
class velocity:
    # u is the velocity along x axis
    # v is the velocity along y axis
    U = 0
    V = 0
    def __init__(self,u,v):
        self.U = u
        self.V = v

# class that contains the particle number, particle centroid position
class frame_2d:
    
    n_particle = 0

    X = []
    Y = []

    def particle_position_random(self,n_part):

        self.n_particle = n_part
        
        self.X = np.random.randint(width, size=n_part)
        self.Y = np.random.randint(height, size=n_part)        

    def assign(self,frame):
        self.X = frame.X.copy()
        self.Y = frame.Y.copy()
        self.n_particle = frame.n_particle # bug here??? shallow copy of n_particle???

    # v is the distribution of velocity
    def move_particle(self,u,v):
        self.X += u
        self.Y += v

        # need to check if out of bound???

# SOM network
class SOM:
    def __init__(self):
        self.layer1 = []
        self.layer2 = []
    
    def fit(self,frame1,frame2,epoch):
        # the third column is the indexes of particles
        self.layer1 = np.zeros([frame1.n_particle,4])
        self.layer2 = np.zeros([frame2.n_particle,4])
        print(self.layer1.shape)
        self.layer1[:,0] = frame1.X
        self.layer1[:,1] = frame1.Y

        self.layer2[:,0] = frame2.X
        self.layer2[:,1] = frame2.Y
        
        self.layer1[:,2] = frame1.X
        self.layer1[:,3] = frame1.Y

        self.layer2[:,2] = frame2.X
        self.layer2[:,3] = frame2.Y

        #coefficient for displacement in each epoch
        alpha = 0.5

        #coefficient to decrease r or increase alpha
        beta = 0.8

        #the radius to determine if the neuron is within a range away from the neuron in the other layer
        r = width / 10

        #threshold to identify pair of particles
        gama = 10
        
        for e in range(epoch):
            if np.less_equal(len(self.layer1),len(self.layer2)):
                length = len(self.layer1)
            else:
                length = len(self.layer2)
            #print(np.linalg.norm(self.layer1[:,0:2]-self.layer2[:,0:2]))
            print('np.mean',np.mean(np.linalg.norm(self.layer1[:,0:2]-self.layer2[:,0:2])))
            mean_error = np.sum(np.mean(np.linalg.norm(self.layer1[:,0:2]-self.layer2[:,0:2])))/length
            
            print('mean_error',mean_error)

            if mean_error > gama:
                # stimulate layer2 with layer 1 first, traverse layer 1 first and update each neuron in layer2
                for i in range(frame1.n_particle):
                   dist = np.linalg.norm(self.layer2[:,0:2] - self.layer1[i,0:2]) #squared or not??
                   min_index = np.argmin(dist)

                   displacement = self.layer1[i,0:2] - self.layer2[min_index,0:2]
                   #modify the neuron weights in the second layer if distance from wc is less than r
                   for j in range(frame2.n_particle):
                       if np.linalg.norm(self.layer2[j,0:2]-self.layer2[min_index,0:2]) <= r:
                           self.layer2[j,0:2] += alpha * displacement

                # should I update r here??

                #stimulate layer1 with layer 2
                for i in range(frame2.n_particle):
                   dist = np.linalg.norm(self.layer1[:,0:2] - self.layer2[i,0:2]) #squared or not??
                   min_index = np.argmin(dist)

                   displacement = self.layer2[i,0:2] - self.layer1[min_index,0:2]
                   #modify the neuron weights in the second layer if distance from wc is less than r
                   for j in range(frame1.n_particle):
                       if np.linalg.norm(self.layer1[j,0:2]-self.layer1[min_index,0:2]) <= r:
                           self.layer1[j,0:2] += alpha * displacement

                r = r * beta
                if alpha / beta <= 1:
                    alpha = alpha / beta
                print('r',r)
                print('alpha',alpha)
            

            plt.subplot(epoch,1,e+1)
            plt.scatter(frame1.X,frame1.Y,color='g',s = 100,marker='o')
            plt.scatter(frame2.X,frame2.Y,color='g',s=100,marker='o')
            plt.scatter(self.layer1[:,0],self.layer1[:,1],color='r',s=50,marker='x')
            plt.scatter(self.layer2[:,0],self.layer2[:,1],color='r',s=50,marker='x')
##            plt.arrow(np.array([frame1.X,frame1.Y]),np.array([frame2.X,frame2.Y])-np.array([frame1.X,frame1.Y]))
            for i in range(len(frame1.X)):
                #plt.arrow(self.layer1[i,0],self.layer1[i,1],self.layer2[i,0]-self.layer1[i,0],self.layer2[i,1]-self.layer1[i,1],head_width=20, head_length=30, fc='r', ec='r')
                plt.arrow(frame1.X[i],frame1.Y[i],frame2.X[i]-frame1.X[i],frame2.Y[i]-frame1.Y[i],head_width=20, head_length=30, fc='k', ec='k')
            plt.title(str(e))

        

def main():
    test = frame_2d()

    test.particle_position_random(100)

    velo = velocity(100,200)
    u = velo.U
    v = velo.V

    test1 = frame_2d()
    test1.assign(test)
    
    test1.move_particle(u,v)
##    plt.scatter(test.X,test.Y,color='g',s = 50,marker='x')
##    plt.scatter(test1.X,test1.Y,color='r',s=50,marker='x')
##    plt.scatter(som.layer1[:,0],som.layer1[:,1],color='g',s=50,marker='o')
##    plt.scatter(som.layer2[:,0],som.layer2[:,1],color='r',s=50,marker='o')
##    plt.title('original')

    
    som = SOM()
    som.fit(test,test1,5)
    print('first X',test.X,'first Y',test.Y)
    print('second X',test1.X,'second Y',test1.Y)
    print('layer1',som.layer1[:,0],som.layer1[:,1])
    print('layer2',som.layer2[:,0],som.layer2[:,1])

    plt.show()

main()
