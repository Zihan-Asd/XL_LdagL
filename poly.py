from logging import exception
import warnings
import numpy as np
import scipy
from scipy.sparse import *
import matplotlib.pylab as plt
import itertools as itertools
import random
from decimal import *

# 
# This file contains classes and methods that will be
# used in other files.
#



class polygen:
    """
    This class is used to generate and extend a polynomial system.
    The ordering of the polynomials is according to (x1,x2,...,xn).
    The generated/extended polynomial system is stored as a
    spasematrix whose rows contain coefficients of polynomial 
    equations.
    ----------
    Attributes
    ----------
    n: number of variables
    D: the highest dimension after extension
    ordering: the ordering is fixed as (1,2,3,...,n) 
    mat_poly: sparse matrix corresponding to the polynomial system.
    ----------
    Methods
    ----------
    extend: extend the degree-2 poly system to degree<=D, a CSC sparse 
     matrix is used to represent the extended poly system.
    extend_list(E): generate a list for all possible monomials with
     degree<= E.
    poly_test: generate a random polynomial with degree=2
    poly_sys_test(n_poly): generate n_poly random polynomials with degree=2
    *testmode_enabled(n_poly): to enable testing. (Only used for testing)
    """
    def __init__(self,n,D):
        self.n = n      #number of variables
        self.D = D      #highest dimension
        self.ordering = np.linspace(1,self.n,num=self.n)    
        # ordering is (x_1,x_2,x_3,...,x_n) by default
        self.polysys = list()

    def extend_list(self,E):
        """
        Extend the degree-2 poly system to degree<=D.
        
        Algorithm:
        -----------
        ****o**o**o**** -> generate all possible polynomials of degree<=D
        """
        self.E = E
        indexset = list(range(self.n+self.E))
        partitions = list(itertools.combinations(indexset,self.n))
        polylist = list()
        for ins in partitions:
            temp = list()
            for i in range(1,self.n):
                temp.append(ins[i]-ins[i-1]-1)
            temp.append(self.n+self.E-ins[self.n-1]-1)
            polylist.append(tuple(temp))
        return polylist

    def poly_test(self):
        """Generate a polynomial equation"""
        poly_list = self.extend_list(2)
        poly = list()
        for ins in poly_list:
            poly.append([random.random()]+list(ins))
        return poly

    def poly_sys_test(self,n_poly):
        """Generate a set of polynomial equations"""
        poly_sys = list()
        for i in range(n_poly):
            poly_sys.append(self.poly_test())
        return poly_sys
    
    def testmode_enabled(self,n_poly):
        self.polysys = self.poly_sys_test(n_poly)
    
    def testmode_example_2(self):
        self.n = 2
        poly_1 = [[1,2,0],[2,1,1],[-3,0,0]]
        poly_2 = [[1,0,2],[3,1,1],[-4,0,0]]
        self.polysys = []
        self.polysys.append(poly_1)
        self.polysys.append(poly_2)

    def testmode_example_3(self):
        self.n = 3
        poly_1 = [[1,2,0,0],[2,1,1,0],[3,1,0,1],[-6,0,0,0]]
        poly_2 = [[1,0,2,0],[3,1,1,0],[1,0,0,2],[-5,0,0,0]]
        poly_3 = [[1,0,0,2],[2,1,0,1],[1,2,0,0],[-4,0,0,0]]
        poly_4 = [[1,1,0,1],[1,0,1,1],[3,1,1,0],[-5,0,0,0]]
        self.polysys = []
        self.polysys.append(poly_1)
        self.polysys.append(poly_2)
        self.polysys.append(poly_3)
        self.polysys.append(poly_4)

    def extend(self):
        polextend_list = self.extend_list(self.D-2)
        poly_list = self.extend_list(self.D)
        poly_list.sort(reverse=True)
        extended_poly = list()
        for poly in self.polysys:
            for ins in polextend_list:
                temp = list()
                for mono in poly:
                    temp_mono = [mono[0]]+list(np.array(mono[1:])+
                    np.array(list(ins)))
                    temp.append(temp_mono)
                extended_poly.append(temp)
        self.len = len(extended_poly)
        self.linvar = len(poly_list)
        poly_i = list()
        poly_j = list()
        poly_data = list()
        for i in range(self.len):
            pol_temp = extended_poly[i]
            for mono in pol_temp:
                j_index = poly_list.index(tuple(mono[1:]))
                poly_i.append(i)
                poly_j.append(j_index)
                poly_data.append(mono[0])
        #print(poly_i)
        #print(poly_j)
        #print(poly_data)
        poly_sparse = csc_matrix((poly_data,(poly_i,poly_j)),shape=(self.len,
        self.linvar),dtype='G')
        poly_sparse.eliminate_zeros()
        #print(poly_sparse)
        return poly_sparse




        




class polysys:
    """
    This class is used to store and process a polynomial system.
    Allowed operations are re-ordering, reduction by assigning values
    to specific variables, gaussian elimination, matrix shape checking,
    matrix shape ploting.
    ----------
    Attributes
    ----------
    mat_poly: sparse matrix corresponding to the polynomial system.
    ordering: the ordering here can be changed. Notice that the 
    polynomial system is always described by a sparsematrix along with
    a certain ordering.
    n: number of variables
    ----------
    datatypes
    ----------
    Sparse matrices are created and stored in COO format and will be 
    converted to CSR/CSC format to enable fast matrix operations 
    such as matrix product, row/column slicing, etc.
    ----------
    Methods
    ----------
    Gauss_elimin: gaussian elimation of a sparse matrix.
    The resulting matrix should look like staircases.
    """
    def __init__(self,mat_poly):
        self.mat_poly = mat_poly       
        # mat_poly here should be a COO format sparse matrix
        
    

    def Gauss_elimin(self):
        # Convert the COO format matrix to CSC format matrix
        # Notice that one should not try to carry out
        # Gaussian elimination on the last column.
        mat_temp_gauss = self.mat_poly
        mat_temp_gauss.eliminate_zeros()
        (n_row,n_col) = mat_temp_gauss.get_shape()
        col_pos = -1
        for i in range(n_row):
            for j in range(col_pos+1,n_col-1):
                coor_nonzeros = mat_temp_gauss.getcol(j).nonzero()[0][:]
                row_coor = -1
                coor_nonzeros_sub = []
                mat_value_sink = []
                for k in coor_nonzeros:
                    if k>(i-1):
                        coor_nonzeros_sub.append(k)
                        mat_value_sink.append(np.abs(mat_temp_gauss.getrow(k).getcol(j
                    ).toarray()[0][0],dtype='g'))
                if len(mat_value_sink) > 0: 
                    max_coor = np.argmax(mat_value_sink)
                    if mat_value_sink[max_coor] > 10**(-16):
                        row_coor = coor_nonzeros_sub[max_coor]
                #for k in coor_nonzeros:
                #    if (k>(i-1) and (abs(mat_temp_gauss.getrow(k).getcol(j
                #    ).toarray()[0][0])>10**(-14))):
                #        row_coor = k
                #        #print(k)
                #        break
                if row_coor >=0:
                    mat_temp_gauss = cscyard(mat_temp_gauss).row_swap(i,row_coor)
                    #print(mat_temp_gauss.toarray())
                    mat_temp_gauss = cscyard(mat_temp_gauss).row_elim(i,j)
                    #print(mat_temp_gauss.toarray())
                    col_pos = j+0-0
                    #print(j)
                    break
                else:
                    # dropping the terms that are too small due to 
                    # numerical instability
                    for k in coor_nonzeros:
                        if k>(i-1):
                            data_k = mat_temp_gauss.getcol(j).getrow(k).toarray()[0][0]
                            dot_mat = csc_matrix(([data_k], ([k],[j])), 
                            shape=(n_row,n_col),dtype='G')
                            mat_temp_gauss = mat_temp_gauss - dot_mat
        #print('Gaussian elimination test')
        #print(mat_temp_gauss.getcol(-1).toarray())
        return mat_temp_gauss
    
    





class polysolver:
    """
    This class is used to solve for an extended polynomial system.
    Two kinds of methods are provided. The first one uses gaussian
    elimination once to obtain a solution for one variable and then
    use this information to reduce the rest of the equations to
    have (n-1) variables. Since the gaussian elimination is already
    done, there is a good chance that we are left with a mono-variate
    polynomial equation for the next variable. In this way, we may 
    obtain the solution for all variables. There is one tricky thing:
    there may be multiple solutions for one variable. We solve this 
    problem by iterative search on a tree using recursion.
    -----------
    Attributes
    -----------
    mat_poly_sys: a polysys class object.
    """

    def __init__(self,mat_gauss,n,D):
        self.mat_gauss = mat_gauss
        self.n = n
        self.D = D  # degree of the extended system
        self.n_temp = self.n
        self.mat_gauss_temp = self.mat_gauss
        [self.n_row_temp,self.n_col_temp] = self.mat_gauss_temp.get_shape()
    

    def shape_check(self):
        """Used to check the shape of the current Gaussian eliminated 
        matrix to make sure there exists a monovariate equation to 
        be solved. If this check fails, higher extension order/ larger 
        number of equations is needed.
        """
        [self.n_row_temp,self.n_col_temp] = self.mat_gauss_temp.get_shape()
        for n_row in range(self.n_row_temp):
            if len(self.mat_gauss_temp.getrow(
                self.n_row_temp-n_row-1).nonzero()[1]):
                index_nonzero = self.mat_gauss_temp.getrow(
                    self.n_row_temp-n_row-1).nonzero()[1][0]
                if index_nonzero < (self.n_col_temp-self.D-1):
                    warnings.warn("No univariate equation is found. "+
                    "Consider a larger extension degree.")
                    return False
                elif index_nonzero < (self.n_col_temp-1):
                    return True
                else:
                    const_ele = self.mat_gauss_temp.getrow(self.n_row_temp
                    -n_row-1).toarray()[0][-1]
                    if abs(const_ele) > 10**(-10):
                        warnings.warn("The poly system has no solution")
                        return False
    
    def single_solve(self):
        pass

    def solve_all(self):
        root = treenode()
        root.n = self.n
        root.D = self.D
        root.mat = self.mat_gauss
        root.level = 0
        global list_sol 
        global grand_level
        grand_level = self.n
        list_sol = []
        treenode.sol_hunt(root)
        #print('test solve_all')
        #print(list_sol)
        return list_sol








class treenode:
    """
    This class is used to construct nodes from a tree for our 
    search algorithm. 
    --------
    Attributes
    --------
    """

    def __init__(self):
        self.key = None
        self.flag = 0   #0 means this node has not been traversed
        self.level = None 
        self.mat = None     #Gauss eliminated matrix, css format
        self.sol = []     # array of solutions
        self.test = []    # array of test solutions
        self.D = None 
        self.n = None   

    def testmode(self):
        self.sol = self.test    #enable test mode
    
    def simplify(self):
        """Used to simplify the GE matrix, and the matrix
        will be updated to self.mat
        """
        poly_list_ori = polygen(self.n+1,self.D).extend_list(self.D)
        poly_list_ori.sort(reverse=True)
        poly_list_sim = polygen(self.n,self.D).extend_list(self.D)
        poly_list_sim.sort(reverse=True)
        counter_sim = 0
        counter_ori = 0
        list_row = []
        list_col = []
        list_data = []
        [n_row_ori,n_col_ori] = self.mat.get_shape()
        n_col_sim = len(poly_list_sim)
        for i in range(n_row_ori):
            if ( len(self.mat.getrow(n_row_ori-1-i).nonzero()[0]) and 
                self.mat.getrow(n_row_ori-1-i).nonzero()[1][0] 
                < (n_col_ori-1-self.D)):
                n_row_sim = n_row_ori - i
                break
        for mono_sim in poly_list_sim:
            flag = 0
            mono_col = csc_matrix((n_row_ori,1),dtype='G')
            for mono_ori in poly_list_ori:
                if mono_sim == mono_ori[:-1]:
                    flag = 1
                    mono_ind_ori = poly_list_ori.index(mono_ori)
                    mono_col = mono_col + self.mat.getcol(mono_ind_ori)*(
                        self.key**(mono_ori[-1])
                    )
                elif flag == 1:
                    break
            mono_ind_sim = poly_list_sim.index(mono_sim)    
            for i in range(len(mono_col.nonzero()[0])):
                if mono_col.nonzero()[0][i] < n_row_sim:
                    list_row.append(mono_col.nonzero()[0][i])
                    list_col.append(mono_ind_sim)
                    list_data.append(mono_col.data[i])
        self.mat = csc_matrix((list_data,(list_row,list_col)),shape=(
            n_row_sim,n_col_sim),dtype='G')
        #print(self.mat.toarray())
        self.mat = polysys(self.mat).Gauss_elimin()
        #plt.figure()
        #cscyard(self.mat).mat_plot()
        #print(self.mat.toarray())
        return self.mat

    def shapecheck(self):
        """Used to check the shape of the current GE matrix"""
        #print(polysolver(self.mat,self.n,self.D).shape_check())
        return polysolver(self.mat,self.n,self.D).shape_check()
    
    def solve(self):
        """Used to obtain an array of possible solutions of a single
        variable
        """
        [n_row,n_col] = self.mat.get_shape()
        #print(n_col)
        uni_poly_list = []
        sol_list = []
        sol_fin = []
        for i in range(n_row):
            if (len(self.mat.getrow(n_row-1-i).nonzero()[0]) and 
                (self.mat.getrow(n_row-1-i).nonzero()[1][0]>(n_col-2-self.D))
                and (self.mat.getrow(n_row-1-i).nonzero()[1][0]<(n_col-1))):
                poly = [0]*(self.D+1)
                for ind in self.mat.getrow(n_row-1-i).nonzero()[1]:
                    poly[ind-(n_col-1-self.D)]=self.mat.getrow(n_row-1-i
                    ).getcol(ind).toarray()[0][0]
                uni_poly_list.append(poly)
            elif (len(self.mat.getrow(n_row-1-i).nonzero()[0]) and 
             (self.mat.getrow(n_row-1-i).nonzero()[1][0]<(n_col-1-self.D))):
                break
        #print(uni_poly_list)
        for i in range(len(uni_poly_list)):
            for j in range(len(uni_poly_list[i])):
                if abs(uni_poly_list[i][j])<10**(-14):
                    uni_poly_list[i][j] = 0
        #print(uni_poly_list)
        temp_list = uni_poly_list.copy()
        for i in range(len(uni_poly_list)):
            for j in range(len(temp_list)):
                if max(list(map(abs,temp_list[j])))<10**(-14):
                    temp_list.pop(j)
                    break
        uni_poly_list = temp_list.copy()
        #print(uni_poly_list)
        for ind_eq in uni_poly_list:
            sol = np.array(np.roots(ind_eq),dtype='G').tolist()
            #print(np.polyval(ind_eq,sol))
            sol_list.append(sol)
        #print(sol_list)
        if len(sol_list):
            for ans in sol_list[0]:
                for j in range(1,len(sol_list)):
                    flag = 0
                    for k in range(len(sol_list[j])):
                        if (abs(ans-sol_list[j][k])<10**(-7)):
                            flag = 1
                    if flag == 0:
                        break
                    if j == (len(sol_list)-1):
                        sol_fin.append(ans)
                if len(sol_list) == 1:
                    sol_fin.append(ans)
        #print(sol_fin)
        if len(sol_fin) == 0:
            #print('False')
            return [False,[]]
        else:
            return [True,sol_fin]

    def purge(self):
        self.key = None
        self.flag = 0   #0 means this node has not been traversed
        self.level = None 
        self.mat = None     #Gauss eliminated matrix, css format
        self.sol = []     # array of solutions
        self.test = []    # array of test solutions
        self.D = None 
        self.n = None 


    
    def sol_hunt(node):
        """
        Used to traverse search for solutions of the poly system.
        """
        global list_sol
        global grand_level
        if node.level < grand_level-1:
            if node.level > 0:
                node.simplify()
            if node.shapecheck() and node.solve()[0]:
                sol_list = node.solve()[1]
                #print(sol_list)
                for ind_sol in sol_list:
                    #print(ind_sol)
                    child_node = treenode()
                    #print(child_node.sol)
                    child_node.key = ind_sol
                    child_node.sol = node.sol
                    #print(node.sol)
                    child_node.sol = node.sol + [ind_sol]
                    #print(node.sol)
                    child_node.n = node.n - 1
                    child_node.D = node.D
                    child_node.mat = node.mat
                    child_node.level = node.level + 1
                    #print(node.level)
                    #print(child_node.sol)
                    treenode.sol_hunt(child_node)
            else:
                return
        else:
            #print(node.level)
            node.simplify()
            if node.solve()[0]:
                sol_list = node.solve()[1]
                for ind_sol in sol_list:
                    #print(node.sol)
                    list_sol.append(node.sol+[ind_sol])
                    #print(list_sol)
                return
            else:
                return
    



class cscyard:
    """
    This class is used to handle sparsematrices of CSC format
    """
    def __init__(self,mat):
        self.mat = mat
        (self.n_row,self.n_col) = self.mat.get_shape()
        self.mat.eliminate_zeros()
    
    def row_swap(self,n,m):
        """swap row n and row m of a CSC matrix"""
        return self.mat-self.row_matrify(n,n)-self.row_matrify(
           m,m)+self.row_matrify(n,m)+self.row_matrify(m,n)

    def row_matrify(self,n,m):
        """Extract row n and make it a matrix whose m-th
        row is the row n and other rows are zero.
        """
        return vstack([csc_matrix((m,self.n_col),dtype='G'),self.mat.getrow(n)
        ,csc_matrix((self.n_row-m-1,self.n_col),dtype='G')
        ])

    
    def row_elim(self,i,j):
        """
        Carry out a Gaussian elimination step using the row r
        whose first non-zero element is at column j.
        """
        coor_nonzeros = self.mat.getcol(j).nonzero()[0][:]
        data = self.mat.getcol(j).getrow(i).toarray()[0][0]
        
        inv_data = 1/data
        
        mat_temp = self.mat
        for k in coor_nonzeros:
            if k>i:
                data_k = mat_temp.getcol(j).getrow(k).toarray()[0][0]
                factor = inv_data*data_k
                mat_temp = mat_temp - self.row_matrify(i,k).multiply(factor)
                data_k = mat_temp.getcol(j).getrow(k).toarray()[0][0]
                dot_mat = csc_matrix(([data_k], ([k],[j])), shape=(self.n_row,
                self.n_col),dtype='G')
                mat_temp = mat_temp - dot_mat
        mat_temp.eliminate_zeros()
        return mat_temp
        
    def mat_plot(self):
        self.mat.eliminate_zeros()
        x_list = self.mat.nonzero()[1]
        y_list = self.mat.nonzero()[0]
        c_list = np.log(np.absolute(np.array(self.mat.data))).tolist()
        plt.scatter(x_list,y_list,c=c_list, s=4,cmap='plasma')
        plt.xlabel("Column number of the coefficient matrix")
        plt.ylabel("Row number of the coefficient matrix")
        cbar = plt.colorbar()
        cbar.set_label("ln(abs(value))")
        plt.gca().invert_yaxis()
        #plt.spy(self.mat,markersize=2)
        


        