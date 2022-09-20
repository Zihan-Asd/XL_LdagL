from poly import *
from Hamiltonian import *
import time


N = 8
a = Hamiltonian_sys(N)
a.fourbody_interaction()
a.threebody_interaction()
a.twobody_interaction()
a.onebody_interaction()
a.simplify()
a.combine()
#coupling = var_rand_list_generator(a.N)
#print(coupling)
coupling = var_list_generator(1,1,1,N)
print("In this simple example,"
    ,"all parameters are set to be 1.")
result = convertor(a.LdagL,coupling,a.N)
start_time = time.time()
b = polysys(result.extend())
k = result.extend()
k.eliminate_zeros()
cscyard(k).mat_plot()
c = b.Gauss_elimin()
plt.figure()
cscyard(c).mat_plot()
sss = polysolver(c,3*N-2,2)
solution = sss.solve_all()
print("The possible solutions of the parameters are",
    solution)
end_time = time.time()
print("Running time for the XL algorithm is ",
    end_time-start_time,"seconds")

print("The first figure corresponds to the coefficient"
    ,"matrix before Gaussian elimination")
print("The second figure corresponds to the coefficient"
    ,"matrix after Gaussian elimination")
