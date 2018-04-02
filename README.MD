This project was developped in order to use autograd compatible Givens Transforms.
Unfortunately, Givens Transforms requires defining roation matrices (which involves 
array assignments) which is incompatible with Autograd. Hence this script was created 
to circumvent this issue so that we may transparently use the givens transform with 
Autograd. In order to use the script, we need the following packages installed:

1) Python 3    
2) Autograd
3) Numba (comes with Anaconda stack)


Givens Transforms is a neat way to transform an optimization problem constrained to
a Stiefel manifold into an unconstrained optimization. Givens Transforms have recently
been exploited for orthonormal inference (https://arxiv.org/pdf/1710.09443.pdf).

I have included a jupyter notebook tutorial going over 2 examples so as to see how
to use the script.