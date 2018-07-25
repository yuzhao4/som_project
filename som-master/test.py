import numpy as np
from som import SOM

# generate some random data with 36 features
data1 = np.random.normal(loc=-.25, scale=0.5, size=(500, 36))
data2 = np.random.normal(loc=.25, scale=0.5, size=(500, 36))
data = np.vstack((data1, data2))

som = SOM(10, 10)  # initialize the SOM
som.fit(data, 2000)  # fit the SOM for 2000 epochs

targets = 500 * [0] + 500 * [1]  # create some dummy target values

# now visualize the learned representation with the class labels
som.plot_point_map(data, targets, ['class 1', 'class 2'], filename='som.png')
som.plot_class_density(data, targets,1, ['class 1', 'class 2'], filename='class_0.png')
