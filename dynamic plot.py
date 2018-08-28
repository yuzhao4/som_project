import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy as np
##
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    pullData = open("samples data.txt","r").read()
    dataArray = pullData.split('\n')
    xar = []
    yar = []
    for eachLine in dataArray:
        if len(eachLine)>1:
            x,y = eachLine.split(',')
            xar.append(int(x))
            yar.append(int(y))
    ax.clear()
    ax.plot(xar,yar)

x0 = np.arange(10)
x1 = np.arange(11,21,1)
print(x0)
print(x1)
    
ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()

