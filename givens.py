import autograd.numpy as np
from numba import jit, f8,i8
from autograd.extend import primitive, defvjp
from numba.types import UniTuple


@primitive
@jit(f8[:,:](f8[:],UniTuple(i8,2)),cache=True)
def theta_2_stiefel(theta,shape):
    n,p = shape
    Z = np.eye(n)[:,:p]   # initialize orthonormal matrix as identity
    idx = -1              # initialize index of theta array
    
    for i in reversed(range(p)):
        for j in reversed(range(i+1,n)):
            # compute cos and sin of rotation angle
            cos = np.cos(theta[idx])
            sin = np.sin(theta[idx])
            # perform rotation on Z matrix
            for k in range(p):
                a = Z[i,k]
                b = Z[j,k]
                Z[i,k] = a*cos+b*sin
                Z[j,k] = -a*sin+b*cos
                      
            idx-=1   # update index of theta
            
    return Z[:,:p]

@jit(f8[:,:,:](f8[:],UniTuple(i8,2)),cache=True)
def theta_2_stiefel_jacobian(theta,shape):
    n,p = shape
    dof = int(n*p-0.5*p*(p+1))
    dZ = np.stack([np.eye(n)[:,:p]]*dof)
    idx = -1
    
    for i in reversed(range(p)):
        for j in reversed(range(i+1,n)):
            # compute cos and sin of rotation angle
            cos = np.cos(theta[idx])
            sin = np.sin(theta[idx])
            for k in reversed(range(dof)):
                if k==idx+dof:
                    dZ_new = np.zeros((n,p))
                    for l in range(p):
                        a = dZ[k,i,l]
                        b = dZ[k,j,l]
                        dZ_new[i,l] = -a*sin+b*cos
                        dZ_new[j,l] = -a*cos-b*sin
                    dZ[k] = dZ_new
                else:
                    # perform rotation on dZ matrix
                    for l in range(p):
                        a = dZ[k,i,l]
                        b = dZ[k,j,l]
                        dZ[k,i,l] = a*cos+b*sin
                        dZ[k,j,l] = -a*sin+b*cos
            idx-=1  # update index
    return dZ

def theta_2_stiefel_vjp(ans,theta,shape):
    dZ = theta_2_stiefel_jacobian(theta,shape)
    return lambda g: np.sum(dZ*g,axis=2).sum(1)

# Tell autograd theta_2_stiefel has custom gradients    
defvjp(theta_2_stiefel, theta_2_stiefel_vjp)




# standard transformation function where theta is constrained between -pi and pi
def transform(theta,shape):
    # constrain theta to be between -pi and pi
    theta = -np.pi + 2*np.pi/(1+np.exp(-theta))
    return theta_2_stiefel(theta,shape)

# Transformation where constraints are specified by user
def transform_constrain(theta,shape,constraints):
    # constrain theta to be between L1 and L2
    L1,L2 = constraints
    theta = L1 + (L2-L1)/(1+np.exp(-theta))
    return theta_2_stiefel(theta,shape)

# Unconstrained transformation
def transform_unconstrain(theta,shape):
    return theta_2_stiefel(theta,shape)


