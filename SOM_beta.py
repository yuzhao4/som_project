import numpy as np
import matplotlib.pyplot as plt
import random

n1 = 100
n2 = 100
DIMENSION = 2

data1 = np.zeros((n1,2))
data2 = np.zeros((n2,2))

for i in range(n1):
    data1[i][0] = np.random.uniform(0,10)
    data1[i][1] = np.random.uniform(0,5)

#print(data1)

for i in range(n1):
    data2[i][0] = np.random.uniform(30,40)
    data2[i][1] = np.random.uniform(0,5)

plt.scatter(data1[:,0],data1[:,1],color='r')
plt.scatter(data2[:,0],data2[:,1],color='g')

def findRange(data):
    l = len(data)

    x = max(data[:,0]) - min(data[:,0])
    y = max(data[:,1]) - min(data[:,1])

    if x > y:
        return x
    else:
        return y

#print('range of data is:',findRange(np.vstack((data1,data2))))

class SOM:
    # there will be m * n neurons, k is dimension of each neuron
    def __init__(self,m,k,data,rand):
        self.m = m
        #self.n = n
        if rand == True:
            self.map = np.random.uniform(0,findRange(data),size = (m,k))
        else:
            self.map = np.zeros((m,k))
            for i in range(m):
                self.map[i][0] = np.random.uniform(15,25)
##            self.map[1][0] = 25
        print('self.map',self.map)

        
    def fit(self,data,epoch):
        trajectories1 = np.zeros((epoch,self.m,2))
##        self.trajectories2 = np.zeros((epoch,m,2))


        c = 0
        alpha = 1
        step = 0.8
        #r = findRange(self.map)+1 #'+1' to avoid dividing 0
        r = 11
        
        for e in range(epoch):
            # iterate each point in data
            trajectories1[e] = self.map

            a = len(data)

##            if e % 1 == 0:
##                c += 1
##                print('c is ',c)
##                plt.subplot(epoch / 10,1,c)
##                plt.scatter(som.map[:,0],som.map[:,1],marker = 'x',s = 50,color = 'g')
##                plt.scatter(data[:,0],data[:,1],color='r')
                #plt.scatter(data2[:,0],data2[:,1],color='g')
            
            for i in range(a):
                #print(data[i]-self.map)
                dist = np.linalg.norm(data[i]-self.map,axis=1)

##                print('dist shape',dist)
                minIndex = np.argmin(dist)
                winner = self.map[minIndex]

                disp = data[i] - winner
                
                #print('disp',disp)
                for j in range(len(self.map)):
                    if np.linalg.norm(self.map[j]-winner) <= r:
                        
                        #alpha = alpha * np.exp(-(np.linalg.norm(self.map[j]-winner)) / r) #???right?? need to normalize???
                        #print('self map before',self.map)
                        self.map[j] += disp * alpha
                        #print('self map after',self.map)
            alpha *= step
            r *= step
            #print('self.map',self.map)
        #print('traj',trajectories1)
        for i in range(self.m):
            plt.plot(trajectories1[:,i,0],trajectories1[:,i,1],label=str(i))
            plt.legend()

data = np.vstack((data1,data2))
#data = data1
som = SOM(4,DIMENSION,data,0)        

np.random.shuffle(data)
print(data)
#print('shuffled data',data)
som.fit(data,1000)
print('map',som.map)
plt.scatter(som.map[:,0],som.map[:,1],marker = 'x',s = 50)

plt.show()
