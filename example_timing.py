from poly import *
from Hamiltonian import *
import time

time_list_all = []
num_trial = 25
N_list = []
parameter_sink = []
for N in range(5,20):
    a = Hamiltonian_sys(N)
    a.fourbody_interaction()
    a.threebody_interaction()
    a.twobody_interaction()
    a.onebody_interaction()
    a.simplify()
    a.combine()
    time_list = []
    parameter_temp = []
    for i in range(num_trial):
        coupling = var_rand_list_generator(a.N)
        parameter_temp.append(coupling)
        result = convertor(a.LdagL,coupling,a.N)
        start_time = time.time()
        b = polysys(result.extend())
        c = b.Gauss_elimin()
        sss = polysolver(c,3*N-2,2)
        sss.solve_all()
        end_time = time.time()
        time_list.append(end_time-start_time)
        print(end_time-start_time)
    time_list_all.append(time_list)
    parameter_sink.append(parameter_temp)
    N_list.append(N)
    print(N)

time_mean = []
time_err = []
for times in time_list_all:
    time_mean.append(np.mean(times))
    time_err.append(np.std(times))

plt.figure()
plt.errorbar(N_list,time_mean,time_err)

