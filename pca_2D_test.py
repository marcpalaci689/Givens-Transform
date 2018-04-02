import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from givens import transform
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D


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
components = 2         # Number of PCA components
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
opt = minimize(pca_objective,theta,method='BFGS',jac=g,args=(cov,shape),options={'gtol':1e-12})

# convert optimal theta to stiefel equivalent
U = transform(opt['x'],shape)

# get the projection plane
x_proj1 = np.arange(-5,6)
x_proj1 = (U[:,:1]*x_proj1).T
x_proj2 = np.arange(-5,6)
x_proj2 = (U[:,1:]*x_proj2).T

xproj = x_proj1+x_proj2[0]
for i in range(1,11):
    xproj = np.vstack((xproj,x_proj1+x_proj2[i]))
    
# Plot training points along with the projection line
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(x[:,0],x[:,1],x[:,2],label='training data')
ax.plot_surface(xproj[:,0].reshape(11,11),xproj[:,1].reshape(11,11),xproj[:,2].reshape(11,11),color='r',alpha=0.3,label='Projection')
ax.grid('off')

# Obtain closed for solution by performing eigendecomposition
e,q = np.linalg.eigh(cov)
U_closed = q[:,-components:][:,::-1]
print('Closed form solution is given as: \n', U_closed)
print('Optimized solution is found as: \n', U,end='\n\n')

# Find the matrix T such that UA = U_closed 
T = np.linalg.lstsq(U,U_closed,rcond=None)[0]

# Assuming this is a rotation matrix, compute the angle and confirm this is indeed a rotation
angle = np.arccos(T[0,0])
rotation = np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
print('T matrix found as: \n', T)
print('rotation matrix associated with an angle of ', np.round(angle,4), ' rad : \n', rotation,end='\n\n')

#Take a step further and solve the problem using a pymanopt to solve the optimization as a constrained problem
from pymanopt import Problem
from pymanopt.solvers import ConjugateGradient
from pymanopt.manifolds import Stiefel

# define objective function
def pca_objective(U):
    return -np.trace(np.dot(U.T,np.dot(cov,U)))

# set up Pymanopt problem and solve.
solver = ConjugateGradient(maxiter=1000)
manifold = Stiefel(D,components)
problem = Problem(manifold=manifold, cost=pca_objective, verbosity=0)
Uopt = solver.solve(problem)

print('Solution found using Pymanopt = \n', Uopt,end='\n\n')

# Find the matrix T such that UA = Uopt 
T = np.linalg.lstsq(U,Uopt,rcond=None)[0]

# Assuming this is a rotation matrix, compute the angle and confirm this is indeed a rotation
angle = np.arccos(T[0,0])
rotation = np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
print('T matrix found as: \n', T)
print('rotation matrix associated with an angle of ', np.round(angle,4), ' rad : \n', rotation)