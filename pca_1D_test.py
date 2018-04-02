import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from givens import transform
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

from pymanopt import Problem
from pymanopt.solvers import SteepestDescent,ConjugateGradient
from pymanopt.manifolds import Stiefel, Euclidean, Product


'''
pca objective function. Theta is a flattened array, cov is an DxD covariance matrix, and shape is a tuple that specifies
the shape of the desired orthonormal matrix.
'''
def pca_objective(theta,cov,shape):
    U = transform(theta,shape)
    return -np.trace(np.dot(U.T,np.dot(cov,U)))

'''
Utilitie to draw 3D samples from a multivariate normal distibution with mean 0 and predefined covariance
'''
def generate_data(N=100):
    cov = np.array([[1.0,0.8,0.3],[0.8,2.,0.2],[0.3,0.2,1.1]])
    x = np.random.multivariate_normal(np.zeros(3),cov,size=N)
    return x

#set constants
N=1000                 # Number of training points
D = 3                  # Dimensionality of training points
components = 1         # Number of PCA components
shape = (D,components) # shape of desired orthonormal matrix
dof = int(D*components-0.5*components*(components+1))  # Degrees of freedom of Stiefel Manifold

# generate data
x = generate_data(N)

#center data
x = x-np.mean(x,axis=0)

#compute covariance
cov = (1.0/N)*np.dot(x.T,x)

# initialize theta such that the steifel manifold is the unit vector along x1 
theta = np.zeros(dof)

# propagates gradients
g = grad(pca_objective)

# Optimize pca_objective with respect to theta
opt = minimize(pca_objective,theta,method='BFGS',jac=g,args=(cov,shape),options={'gtol':1e-8})

# convert optimal theta to stiefel equivalent
U = transform(opt['x'],shape)

# get projection line 
x_proj = np.arange(-5,6)
x_proj = (U*x_proj).T

# Plot training points along with the projection line
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(x[:,0],x[:,1],x[:,2],label='training data')
ax.plot(x_proj[:,0],x_proj[:,1],x_proj[:,2],c='r',alpha=0.4,lw=3,label='Projection')
ax.legend()

# Obtain closed for solution by performing eigendecomposition
e,q = np.linalg.eigh(cov)
U_closed = q[:,-components:][:,::-1]
print('Closed form solution is given as: \n', U_closed)
print('Optimized solution is found as: \n', U)


