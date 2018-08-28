import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy as np

fig1, ax = plt.subplots()

data0 = np.loadtxt("1.txt", delimiter=",")
data1 = np.loadtxt('2.txt',delimiter=',')

x0 = data0[:,0]
y0 = data0[:,1]

x1 = data1[:,0]
y1 = data1[:,1]

line1, = ax.plot([],[],'-',label = '0')
line2, = ax.plot([],[],'--',label='1')

ax.set_xlim(np.min(x0),np.max(x0))
ax.set_ylim(np.min(y0),np.max(y0))
ax.set_xlim(np.min(x1),np.max(x1))
ax.set_ylim(np.min(y1),np.max(y1))
def worker():

    ani = animation.FuncAnimation(fig1, animate, frames = 1000,interval=10)
    plt.legend()
    plt.show()
    
def animate(i):
    line1.set_xdata(x0[:i])
    line1.set_ydata(y0[:i])
    line2.set_xdata(x1[:i])
    line2.set_ydata(x1[:i])
    return line1,line2

worker()
##    pullData0 = open("./1.txt","r").read()
##    pullData1 = open("./2.txt","r").read()
##
##    dataArray0 = pullData0.split('\n')
##    dataArray1 = pullData1.split('\n')
##    
##    xar0 = []
##    yar0 = []
##
##    xar1 = []
##    yar1 = []
##
##    for eachLine in dataArray0:
##        if len(eachLine)>1:
##            x,y = eachLine.split(',')
##            xar0.append(int(float(x)))
##            yar0.append(int(float(y)))
##
##    for eachLine in dataArray1:
##        if len(eachLine)>1:
##            x,y = eachLine.split(',')
##            xar1.append(int(float(x)))
##            yar1.append(int(float(y)))
##
##    ax0.clear()
##    ax1.clear()
##    
##    ax0.plot(xar0,yar0)
##    ax1.plot(xar1,yar1)
##def first():
##    
##    with open('./1.txt','a') as firstFile:
##
##        for i in range(1000):
##            firstFile.write(str(i) + ',' + str(i + 10) + '\n')
##
##def second():
##    with open('./2.txt','a') as secondFile:
##        for i in range(1000):
##            secondFile.write(str(i) + ',' + str(i - 10) + '\n')
##
##import threading
###threads = []
##
##t0 = threading.Thread(target = worker)
##t1 = threading.Thread(target = first)
##t2 = threading.Thread(target = second)
##
##t0.start()
##t1.start()
##t2.start()
