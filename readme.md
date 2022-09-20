XL_LdagL is used to solve for the parameters of L after 
assigning LdagL with some operator value. This package
consists of two files, the poly.py file and the 
Hamiltonian.py file.
The core file 
is the poly.py file which contains classes to handle 
polynomial systems and to solve for degree-2 polynomial
systems using the XL algorithm.[1] The Hamiltonian.py file
generates the LdagL using a particular L model-XXZ model
under decay. 


The example_Gaussian_elim.py file illustrates how to use
the package to solve for a particular LdagL generated 
from a XXZ model under decay with all parameters set to 1. 


The example_timing.py file illustrates the XL running time
to solve for LdagLs generated from XXZ models under decay with random parameters. 


[1]N. Courtois, A. Klimov, J. Pararin, and A. Shamir. Efficient Algorithms for
Solving Overdefined Systems of Multivariate Polynomial Equations. In B. Preneel,
editor, Advances in Cryptology — EUROCRYPT 2000, volume 1807 of LNCS,
pages 392–407. Springer-Verlag, Berlin, 2000