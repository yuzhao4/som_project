import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(1,5,5,endpoint = True)
degree = np.linspace(5,15,5,endpoint = True)

som_accu_linear = [0.26,0.65,0.99,1.0,1.0]
nb_accu_linear = [0.2,0.64,0.88,0.95,0.97]

som_accu_rotate = [0.86,0.56,0.44,0.48,0.46]
nb_accu_rotate = [0.73,0.54,0.33,0.21,0.11]

som_accu_ansys = [0.9375,0.9583,0.9583,0.9326,0.8194]
nb_accu_ansys = [0.0972,0.0243,0.0347,0.0416,0.059]

plt.plot(x,som_accu_linear,c='r',marker = 'o',label='som')
plt.plot(x,nb_accu_linear,c='g',marker = 'x',label='nb')
plt.xlabel('lmbda')
plt.ylabel('Accuracy')
plt.title('uniform velocity comparison')
plt.legend()
plt.show()

##plt.plot(degree,som_accu_rotate,c='r',marker = 'o',label='som')
##plt.plot(degree,nb_accu_rotate,c='g',marker = 'x',label='nb')
##plt.xlabel('degree')
##plt.ylabel('Accuracy')
##plt.title('rotation comparison')
##plt.legend()
##plt.show()

plt.plot([0,0.16,0.32,0.48,0.64],som_accu_ansys,c='r',marker = 'o',label='som')
plt.plot([0,0.16,0.32,0.48,0.64],nb_accu_ansys,c='g',marker = 'x',label='nb')
plt.xlabel('Time')
plt.ylabel('Accuracy')
plt.title('Simulation particles comparison')
plt.legend()
plt.show()
