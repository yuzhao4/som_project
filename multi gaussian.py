import numpy as np
import matplotlib.pyplot as plt

##mean = [0, 0]
##cov = [[1, 10], [10, 30]]  # diagonal covariance
##
##x, y = np.random.multivariate_normal(mean, cov, 500).T
##plt.plot(x, y, 'x')
##plt.axis('equal')

from scipy.stats import multivariate_normal
x = np.linspace(0, 5, 100, endpoint=False)
y = multivariate_normal.pdf(x, mean=2.5, cov=0.5); y

print('sum of f pdf',np.sum(y,axis=0) * 5 / 100)


plt.plot(x, y)

##x, y = np.mgrid[-1:1:.01, -1:1:.01]
##print('x shape',x.shape)
###print('y',y)
##pos = np.empty(x.shape + (2,))
##print('pos shape',pos.shape)
##pos[:, :, 0] = x; pos[:, :, 1] = y
####print(pos)
##rv = multivariate_normal([0, 0], [[3, 1], [1, 3]])
##print('pdf',rv.pdf(pos))
##print('sum of pdf',np.sum(rv.pdf(pos)))
##plt.contourf(x, y, rv.pdf(pos))

plt.show()
