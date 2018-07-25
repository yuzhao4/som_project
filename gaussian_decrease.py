import numpy as np
import matplotlib.pyplot as plt
# this function is to generate a series of decreasing number from a gaussian distribution

a = []

alpha_start = 0.2
sigma = 1
mean = 0

x = np.linspace(0,1,50)

for i in x:
    a.append(alpha_start * np.exp(-(i-mean)**2 / sigma**2))
    
plt.scatter(x,a)
plt.show()
print(a)

# where you are gonna use this??
# 1st when decaying the radius of searching circle
# 2nd when decaying the coefficient for moving distance

# can I use the same distribution??
# are they related???

# radius is used to move the neurons within a area near the winner neuron
# the coefficient is used to move a fraction distance between the neuron and the nearest data point
