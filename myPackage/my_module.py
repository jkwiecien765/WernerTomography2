# Functions for Werner app
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections.abc import Iterable
from scipy.linalg import sqrtm
from numpy import pi
from .__init__ import parameters


np.set_printoptions(precision=5,suppress=True)
     


def chop(expr, max=1e-15):
    
    expr = np.asarray(expr) if type(expr)==np.matrix else expr
    if issubclass(type(expr), Iterable):
        return [chop(i) for i in expr]
    else:
        return (expr if expr**2 > max**2 else 0.0)
    

#Statevectors 1 and 2 qubit
Zero=np.array([0,1])
One=np.array([1,0])
ZeroZero=np.array([0,0,0,1])
ZeroOne=np.array([0,0,1,0])
OneZero=np.array([0,1,0,0])
OneOne=np.array([1,0,0,0])

#Pauli matrices
Pauli = (np.array([[1,0],[0,1]], dtype=complex), np.array([[0,1],[1,0]], dtype = complex), 
         np.array([[0, 0+1j],[0-1j, 0]], dtype= complex), np.array([[1,0],[0,-1]], dtype=complex))
#Antidiagonal identity matrix
F = np.array([[0,0,0,1],
                [0,0,1,0],
                [0,1,0,0],
                [1,0,0,0]])

#Tensor product which doesn't contain nested list in output
def tens_prod2d(u1,u2):

    U=np.tensordot(u1,u2,0)
    ua=np.concatenate((U[0][0],U[0][1]),1)
    ub=np.concatenate((U[1][0],U[1][1]),1)
    u3=np.concatenate((ua,ub),0)
    return np.array(u3)

def unitary_mat_par(params):
    th = params[0]
    alpha = params[1]
    beta = params[2]
    u1=np.array([[np.exp(1j* alpha)*np.cos(th), np.exp(1j* beta)*np.sin(th)],\
                    [-np.exp(-1j* beta)*np.sin(th),np.exp(-1j* alpha)*np.cos(th)]])
    return u1

def unitary_mat2(params=None):
    
    if(params!=None):
        th = params[0]
        alpha = params[1]
        beta = params[2]
        u1=np.array([[np.exp(1j* alpha)*np.cos(th), np.exp(1j* beta)*np.sin(th)],\
                        [-np.exp(-1j* beta)*np.sin(th),np.exp(-1j* alpha)*np.cos(th)]])
        return u1
    else:
        from scipy import stats
        mat = stats.unitary_group.rvs(2)
        mat = mat / np.power(np.linalg.det(mat), 1/2)
        return np.array(mat)


def unitary_mat3(L=2):
    from scipy import stats
    mat = stats.unitary_group.rvs(L)
    mat = mat / np.power(np.linalg.det(mat), 1/L)
    return np.array(mat)

#Density matrix of Werner state and its generalisation
def rho2(th, vis):
    '''returns 4x4 ndarray matrix of Werner state
    first argument is angle of sine in the formula, second is the visibility'''
     
    entgl=np.sin(th)*ZeroOne+np.cos(th)*OneZero
    return vis * np.outer(entgl,entgl)\
          + (1-vis)/4 * np.identity(4)

#Produces 3 random numbers [0, 2Pi]

def rand_phase():
    r=[np.arcsin(np.sqrt(np.random.rand())),np.random.rand()*2*pi,np.random.rand()*2*pi]
    return r

parameters=[]
for i in range(1000000):
    parameters.append([rand_phase(),rand_phase()])
    
def aT(matrix):
    '''Transposes a 4D matrix over its antidiagonal '''
    matrix = F@matrix.T@F
    return matrix 

def rotate_matrix_par(matrix, paramsA, paramsB):
    matrix = matrix.matrix if type(matrix) == density_matrix else matrix
    uA = unitary_mat_par(paramsA)
    uB = unitary_mat_par(paramsB)
    uAB = tens_prod2d(uA, uB)
    return np.transpose(np.conjugate(uAB))@matrix@uAB 


def rotate_matrix(matrix, paramsA=None, paramsB=None):
    matrix = matrix.matrix if type(matrix) == density_matrix else matrix
    uA = unitary_mat2(paramsA)
    uB = unitary_mat2(paramsB)
    uAB = tens_prod2d(uA, uB)
    return np.transpose(np.conjugate(uAB))@matrix@uAB   

def obs(rho,parA = None, parB = None):
    '''Simulation of observation of density matrix with unitary matrices of given parameters (defaults to random) 
        returns probability of observation as being in 00 state'''
    uA = unitary_mat2(parA)
    uB = unitary_mat2(parB)    
    u = tens_prod2d(uA,uB)
    zer = np.outer(ZeroZero,ZeroZero)
    p = rho@(np.transpose(np.conjugate(u)))@zer@u
    return np.real(np.trace(p))



def classical_fidelity(binsA, binsB, N=100):
    '''Calculates classical fidelity given two np.arrays of bin counts (or density_matrices)'''
    if(type(binsA) == density_matrix):
        binsA = binsA.bins(N)['counts']
    if(type(binsB) == density_matrix):
        binsB = binsB.bins(len(binsA))['counts']
    
    if(len(binsA)!=len(binsB)):
        raise ValueError("Bins must be of the same lenght")
    
    return np.sum(np.sqrt(binsA*binsB))**2

def classical_fidelity2(binsA, binsB, N=100):
    '''Calculates classical fidelity given two np.arrays of bin counts (or density_matrices)'''
    if(type(binsA) == density_matrix):
        binsA = binsA.bins(N)['counts']
    if(type(binsB) == density_matrix):
        binsB = binsB.bins(len(binsA))['counts']
    
    if(len(binsA)!=len(binsB)):
        raise ValueError("Bins must be of the same lenght")
    cf=0
    mins = np.min([binsA, binsB], axis=0)
    cf = np.sum(mins)
    return cf

def classical_fidelity3(dmA, dmB):
    probs = dmA.bins()['bins']
    probs = probs + probs[1]/2     #setting each value to be in the middle of intervals
    arrA = dmA.data/len(dmA.data)*4
    arrB = dmB.data/len(dmB.data)*4
    Len = min(len(dmA.data), len(dmB.data))
    fid = np.power(np.sqrt(arrA[:Len]*arrB[:Len]).sum(), 2)
    return fid

def double_plot(dmA, dmB):
    if type(dmA)==density_matrix:
        binsA = dmA.bins()['counts']
    else:
        binsA = dmA
    if type(dmB)==density_matrix:
        binsB = dmB.bins()['counts']
    else:
        binsB = dmB
    bins = np.linspace(0,1,101)
    plt.stairs(binsA, bins, fill=True)
    plt.stairs(binsB, bins)
    
def matrix_fidelity(matrixA, matrixB):
    '''Calculates matrix fidelity given two ndarrays of matrices (or density_matrices)'''

    if(type(matrixA) == density_matrix):
        matrixA = matrixA.matrix
    if(type(matrixB) == density_matrix):
        matrixB = matrixB.matrix

    sqr = sqrtm(matrixA)
    if np.isnan(sqr).any():
        print('Faulty Matrices:', matrixA, matrixB)
        return np.nan
    
    fid = min(np.real(np.trace(sqrtm(sqr@matrixB@sqr)))**2,1)
    
    return fid

def optimal_matrix_fidelity(dmA):
    from scipy.optimize import differential_evolution
    def f(params, matrixA):
        matrixA = matrixA.matrix if type(matrixA) == density_matrix else matrixA
        matrixB = rho2(params[-1], 1)
        paramsA = params[:3]
        paramsB = params[3:-1]
        
        return -1*matrix_fidelity(matrixA,rotate_matrix(matrixB, paramsA, paramsB))
    bounds = [(0,2*pi), (0,2*pi), (0,2*pi), (0,2*pi), (0,2*pi), (0,2*pi), (0, pi/4)]
    res = differential_evolution(f, args=(dmA,), bounds=bounds)
    return {'value': -res['fun'], 'angle': res['x'][-1], 'parameters': [res['x'][:3].tolist(), res['x'][3:6].tolist()]}

def optimal_matrix_fidelity_vis(dmA):
    from scipy.optimize import differential_evolution
    def f(params, matrixA):
        matrixA = matrixA.matrix if type(matrixA) == density_matrix else matrixA    #inspected matrix
        matrixB = rho2(params[-2], params[-1])
        paramsA = params[:3]
        paramsB = params[3:6]
        return Frobenius_dist(rotate_matrix(matrixB, paramsA, paramsB), matrixA)
    bounds = [(0,2*pi), (0,2*pi), (0,2*pi), (0,2*pi), (0,2*pi), (0,2*pi), (0, pi), (0,1)]
    res = differential_evolution(f, args=(dmA,), bounds=bounds)
    return {'distance': res['fun'], 'angle': res['x'][-2], 'visibility': res['x'][-1], 'parameters': [res['x'][:3].tolist(),res['x'][3:6].tolist()]}

def optimal_matrix_rotation(dmA, dmB):
    from scipy.optimize import differential_evolution
    def f(params, matrixA, matrixB):
        matrixA = matrixA.matrix if type(matrixA) == density_matrix else matrixA
        matrixB = matrixB.matrix if type(matrixB) == density_matrix else matrixB
        paramsA = params[:3]
        paramsB = params[3:]
        return -1*matrix_fidelity(rotate_matrix(matrixA, paramsA, paramsB), matrixB)
    res = differential_evolution(f, args=(dmA, dmB), bounds=[(0,2*pi), (0,2*pi), (0,2*pi), (0,2*pi), (0,2*pi), (0,2*pi)])
    return {'value': -res['fun'], 'parameters': [res['x'][:3],res['x'][3:]]}

def compare_fid(dmA, dmB, show_fidelity = False):
    print(f'Matrix fidelity {matrix_fidelity(dmA, dmB):.4f}, \n optimal matrix fidelity: {optimal_matrix_fidelity(dmA, dmB)["value"]:.4f}, \n geometrical classical fidelity {classical_fidelity2(dmA, dmB)[0]:.4f}, \n statistical bin classical fidelity {classical_fidelity(dmA, dmB):.4f}, \n statistical point classical fidelity {classical_fidelity3(dmA, dmB):.4f}')
    binsA = dmA.bins()
    binsB = dmB.bins()
    plt.stairs(binsA['counts'], binsA['bins'], fill=True)
    plt.stairs(binsB['counts'], binsB['bins'])
    if show_fidelity:
        mins = classical_fidelity2(dmA, dmB)[1]
        plt.stairs(mins, binsB['bins'], color = 'black')
    plt.show()



def Frobenius_dist(A, B):
    '''Frobenius distance of two states. Input must me two 4x4 matrices or density_matrices'''
    A=A.matrix if type(A)==density_matrix else A
    B=B.matrix if type(B)==density_matrix else B
        
    D=A-B
    dist=np.sqrt(np.real(np.trace(np.transpose(np.conjugate(D))@D)))
    return dist
    


def vis_optimizer_dm(dm2, dm1, plot=False, N=200, printing=True):
    '''Optimises a state's (2nd argument) visibility with respect to experimental data in bins (or density_matrix or simple 4x4 ndarray)'''
    mf=matrix_fidelity(dm2, dm1)

    dist=Frobenius_dist(dm2, dm1)
    if printing:
        print(f'Initial distance {dist}')
        print(f'Initial matrix fidelity: {mf}')
    
    if(mf < 0.25):
        vis = 0.0
    else:
        vis = 1.3333333333333333333 * (mf - 0.25)
    if printing:
        print(f'Optimal visibility: {vis}')
    opt_matrix=dm1*vis+(1.0-vis)*density_matrix(np.diag([0.25,0.25,0.25,0.25]))

    mf=matrix_fidelity(dm2, opt_matrix)
    dist=Frobenius_dist(dm2, opt_matrix)
    if printing:
        print(f'Final matrix fidelity: {mf}')
        print(f'Final distance: {dist}')
        
    if(plot):
        plt.hist(dm2.data,bins=N,density=True)
        #plt.hist(dm1.data,bins=N, density=True, histtype='step')
        if(opt_matrix.data == []): 
            opt_matrix.set()
        plt.hist(opt_matrix.data, bins=N, histtype='step', density = True)
        plt.show()
    
    return opt_matrix, vis


#Copied from qiskit
def _ginibre_matrix(nrow, ncol, seed=None):
    """Return a normally distributed complex random matrix.

    Args:
        nrow (int): number of rows in output matrix.
        ncol (int): number of columns in output matrix.
        seed(int or np.random.Generator): default rng.

    Returns:
        ndarray: A complex rectangular matrix where each real and imaginary
            entry is sampled from the normal distribution.
    """
    if seed is None:
        rng = np.random.default_rng()
    elif isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    ginibre = rng.normal(
        size=(nrow, ncol)) + rng.normal(size=(nrow, ncol)) * 1j
    return ginibre

#Copied from qiskit
def rand_PSDM(dim, seed=None):
    """
    Generate a random density matrix from the Hilbert-Schmidt metric.

    Args:
        dim (int): the dimensions of the density matrix.
        rank (int or None): the rank of the density matrix. The default
            value is full-rank.
        seed (int or np.random.Generator): default rng.

    Returns:
        ndarray: rho (N,N)  a density matrix.
    """
    mat = _ginibre_matrix(dim, dim, seed)
    mat = mat.dot(mat.conj().T)
    return mat / np.trace(mat)
        
def mean_over_unitars(matrix, N=100000, recording=False):
    '''Takes a matrix or 4x4 list/ndarray and translates it N times over unitary matrices. If recording=True, it returns also a pandas.DataFrame with each iteration of the loop'''
    matrix=(matrix.matrix if type(matrix)==density_matrix else matrix)
    record=pd.DataFrame()
    for i in range(N):
        if recording:
            ser=pd.Series(np.append(np.asarray(matrix).flatten(), np.trace(matrix))).to_frame().T
            record=pd.concat([record, ser])    
        uA=np.array(unitary_mat2())
        uB=np.array(unitary_mat2())
        u=tens_prod2d(uA,uB)
        matrix = u@matrix@(np.transpose(np.conjugate(u)))
        #matrix /= np.trace(matrix)
    if recording:
        record.reset_index(inplace=True)
        record.drop('index', axis=1, inplace=True)
        record.rename({k: str(k) for k in range(16)}, axis=1, inplace=True)
        record.rename({16: 'Trace'}, axis=1, inplace=True)            
        return matrix, record
    else:
        return matrix    

def mean_over_unitars2(initial_matrix, N=100000, recording=False):
    '''Takes a matrix or 4x4 list/ndarray and takes average of N translations over unitary matrices. If recording=True, it returns also a pandas.DataFrame with each iteration of the loop'''
    initial_matrix = np.array(initial_matrix)
    final_matrix = np.array(np.zeros([4,4]), dtype='complex128')
    record = pd.DataFrame()
    matrix=final_matrix
    ser=pd.Series(np.asarray(matrix).flatten()).to_frame().T
    record=pd.concat([record, ser])
    
    for i in range(N):        
        uA=np.array(unitary_mat2())
        u=tens_prod2d(uA,uA)
        final_matrix += u@initial_matrix@(np.transpose(np.conjugate(u)))/N
        if recording:
            matrix=final_matrix*N/len(record)
            ser=pd.Series(np.asarray(matrix).flatten()).to_frame().T
            record=pd.concat([record, ser])
        #matrix /= np.trace(matrix)
    if recording:
        record.reset_index(inplace=True)
        record.drop('index', axis=1, inplace=True)
        record.rename({k: str(k) for k in range(16)}, axis=1, inplace=True)
        record.rename({16: 'Trace'}, axis=1, inplace=True)            
        return final_matrix, record
    else:
        return final_matrix   

'''MEASURES'''

def concurrence(dm):
    rho = dm.matrix if type(dm)==density_matrix else np.array(dm)  #making sure rho is of np.array type
    rhod = tens_prod2d(Pauli[2], Pauli[2])@np.conjugate(rho)@tens_prod2d(Pauli[2], Pauli[2])
    lambs = np.linalg.eigvals(rho@rhod)
    lambs = np.sqrt(lambs)
    l1 = max(lambs)
    C = max(0, 2*l1 - np.sum(lambs))
    return np.real(C)
     
def correlation_matrix(dm):
    rho = dm.matrix if type(dm)==density_matrix else dm #making sure rho is of np.array type
    T=np.zeros((3,3), dtype=complex)
    for i in range(3):
        for j in range(3):
            T[i][j] = np.trace(rho@tens_prod2d(Pauli[i+1], Pauli[j+1])) #Pauli[0] is identity
    return np.array(np.real(T))

def CHSHviolation_measure(dm):
    rho = dm.matrix if type(dm)==density_matrix else dm  #making sure rho is of np.array type
    T = correlation_matrix(rho)
    U = T.transpose()@T
    lambs = np.linalg.eigvals(U)
    lambs = np.sort(lambs)
    M = lambs[-1] + lambs[-2]
    B = np.sqrt(min(max(0, M-1),1))
    return B


'''MAIN CLASS'''
    
class density_matrix:
    def __init__(self, rho, name=''):
        if np.shape(rho)!=(4,4):
            raise TypeError("Density matrix must be 4x4 array")
        self.matrix=np.array(rho)
        self.name=name
        
    def __str__(self):
        return self.name
            
    def __add__(self, density_matrix2):
        return(density_matrix(self.matrix+density_matrix2.matrix))
    
    def __mul__(self, num):
        if not isinstance(num, float):
            try:
                num = float(num)
            except:
                raise TypeError("You must multiply by a float")
            
        return(density_matrix(num*self.matrix))
            

    def __rmul__(self, num):
        if not isinstance(num, float):
            try:
                num = float(num)
            except:
                raise TypeError("You must multiply by a float")
            
        return(density_matrix(num*self.matrix))
    
    
    def set(self,N=50000, start=0):
        self.data=[]
        params=parameters[start:start+N]
        for i in range(len(params)):
            self.data.append(obs(self.matrix,params[i][0],params[i][1]))
        
        self.data = np.array(self.data)
    
        
    desc=""
    is_Werner=False
    num_of_compounds=np.nan
    Werner_angle=[]
    weights=[]
    visibility=np.nan
    data=[]
    name=''
    
    def aT(self):
        return density_matrix(aT(self.matrix))    
     
    def T(self):
        return density_matrix(self.matrix.T)    
    
    def Ur(self, paramsA, paramsB):
        return density_matrix(rotate_matrix(self.matrix, paramsA, paramsB))
    
    def range(self):
        if len(self.data)==0:    
            #print("setting density_matrix data...")
            self.set() 
        mi=np.round(min(self.data),3)
        ma=np.round(max(self.data),3)
        return(mi, ma)
    
    def histogram(self, BinNum=100, AdjustBins=False):
        if len(self.data)==0:    
            #print("setting density_matrix data...")
            self.set()
            
        if(AdjustBins):
            ran=max(self.range()[1]-self.range()[0],0.001)
            plt.hist(self.data,int(BinNum/ran),range=(0,1),density=True)
        else:
            plt.hist(self.data,BinNum,range=(0,1),density=True)
    
    def bins(self, BinNum=100, AdjustBins=False):
        if len(self.data)==0:    
            #print("setting density_matrix data...")
            self.set() 
        n=BinNum
        if(AdjustBins):
            ran=max(self.range()[1]-self.range()[0],0.001)
            bin=ran/BinNum
            n=n/ran
            bins=np.linspace(0,1,int(1/bin)+1)
            counts=np.zeros(int(1/bin))
        else:
            bins=np.linspace(0,1,BinNum+1)
            counts=np.zeros(BinNum)
        for dat in self.data:
            try:
                counts[int(dat*BinNum)]+=1/len(self.data)
            except IndexError:
                pass
        Bins={
            "counts" : counts,
            "bins" : bins
        }
        return Bins
    
    def curve(self):
        Bins=self.bins()
        counts=Bins['counts']
        bins=Bins['bins'][1:]
        counts2=counts[:2]
        for idx in range(len(counts)-6):
            count=counts[idx+3]
            count2=0
            denum=0
            for delta in range(7):
                if count/counts[idx+delta]<2 and count/counts[idx+delta]>0.5:
                    denum+=1
                    count2+=counts[idx+delta]
            if count2>0:
                count2/=denum 
            counts2=np.append(counts2,count2)
        for el in counts[-4:]:
            counts2=np.append(counts2,el)
        plt.plot(bins,counts2)
        return


def hist_convolution(binsf,binsg):
    if(len(binsf)!=len(binsg)):
        raise ValueError("Unmatched bins set! (different lengths)")
    l=len(binsf)
    conv_bins=[]
    for i in range(l):
        conv=0
        for j in range(i+1):
            conv+=binsf[i-j]*binsg[j]*l
        conv_bins.append(conv)
    return conv_bins 


"""Calculate the Brues distance.
Input: two density matrices (array form)
Output: Mean distance: float [0,1]"""
def loss_function(dms1, dms2):
    loss=0
    for i in range(len(dms1)):
        dm1=dms1[i]
        dm2=dms2[i]
        fidelity=min(np.trace(sqrtm(dm1)@dm2@sqrtm(dm1))**2,1)
        loss+=2-2*np.sqrt(fidelity)
    return loss


def bins2curve(Bins):
    counts=Bins['counts']
    bins=Bins['bins'][1:]
    counts2=counts[:2]
    for idx in range(len(counts)-6):
        count=counts[idx+3]
        count2=0
        denum=0
        for delta in range(7):
            if count/counts[idx+delta]<2 and count/counts[idx+delta]>0.5:
                denum+=1
                count2+=counts[idx+delta]
        if count2>0:
            count2/=denum 
        counts2=np.append(counts2,count2)
    for el in counts[-4:]:
        counts2=np.append(counts2,el)
    plt.plot(bins,counts2)
    return
    

'''Data generation and pre-processing'''

def data_generator(dm=None):
    dm = density_matrix(rand_PSDM()) if dm==None else dm
    dm.name = 'rand_PSDM'
    try:
        ans = optimal_matrix_fidelity(dm)
    except:
        return {}
    angle = np.real(ans['angle'])
    rotation = np.real(ans['parameters'])
    opt_matrix, vis = vis_optimizer_dm(dm, density_matrix(rotate_matrix(rho2(angle, 1), rotation[0], rotation[1])), printing = False)
    opt_matrix.name = 'angle=' + str(angle) + ', vis=' + str(vis)
    opt_matrix.matrix = opt_matrix.matrix/np.trace(opt_matrix.matrix)
    hist = np.real(dm.bins()['counts']).tolist()
    
    return {'Matrix': dm.matrix.tolist(), 'Bins': hist, 'Angle': angle, 'Visibility': np.real(vis), 'Rotation': rotation,
            'Distance': np.real(Frobenius_dist(dm, opt_matrix)), 'MatrixFidelity': np.real(matrix_fidelity(dm, opt_matrix)),
            'HistogramFidelity': np.real(classical_fidelity(dm, opt_matrix)), 'Covering': np.real(classical_fidelity2(dm, opt_matrix)),
            'ConcurrenceOriginal': np.real(concurrence(dm)), 'ConcurrenceOpt': np.real(concurrence(opt_matrix)), 
            'CHSHViolationMOriginal': np.real(CHSHviolation_measure(dm)), 'CHSHViolationMOpt': np.real(CHSHviolation_measure(opt_matrix))}

def data_order(dictionary):
    bins = np.linspace(0,1,101)
    bins2 = []
    for i in range(100):
        bins2.append('[' + str(round(bins[i],2)) + ', ' + str(round(bins[i+1],2)) + ']')
    binsDF = pd.DataFrame({0: dictionary['Bins'], 'Index': bins2}).set_index(['Index']).transpose()
    binsDF = binsDF

    matrixList = []
    matrixIndex = []
    for i in range(4):
        for j in range(4):
            matrixList.append(dictionary['Matrix'][i][j])
            matrixIndex.append(str(i)+','+str(j))
    matrixDF = pd.DataFrame({'Index': matrixIndex, 0: matrixList}).set_index(['Index']).transpose()
    matrixDF = matrixDF
    
    rotationList = []
    rotationIndexl0 = []
    for i in range(2):
        for j in range(3):
            rotationList.append(dictionary['Rotation'][i][j])
            rotationIndexl0.append(3*i+j)
    rotationDF = pd.DataFrame({'Index': rotationIndexl0, 0: rotationList}).set_index(['Index']).transpose()

    paramsList = [dictionary['Angle'], dictionary['Visibility']]
    paramsIndexl0= ['Angle', 'Visibility']
    paramsDF = pd.DataFrame({'Index': paramsIndexl0, 0: paramsList}).set_index(['Index']).transpose()
    
    measuresIndexl0 = ['Distance',  'MatrixFidelity', 'HistogramFidelity', 'Covering', 'ConcurrenceOriginal', 'ConcurrenceOpt', 'CHSHViolationMOriginal', 'CHSHViolationMOpt']
    measuresList = [dictionary[key] for key in measuresIndexl0]
    measuresDF = pd.DataFrame({'Index': measuresIndexl0, 0: measuresList}).set_index(['Index']).transpose()
        
    return [binsDF, matrixDF, rotationDF, paramsDF, measuresDF]


def data_saver(name, n=1000):
    
    categories = ('Bins', 'Matrix','Rotation', 'OptimalState', 'Measures')     
    dfs = data_order(data_generator())
    for i in range(n-1):
        t0=time.time()
        for j, df in enumerate(data_order(data_generator())):
            dfs[j] = pd.concat((dfs[j], df))
        deltat = time.time() - t0
        print(f'Successfuly simulated {i+1} of {n} samples. Time elapsed: {deltat:.2f}')
    for i, df in enumerate(dfs):
        df = df.reset_index(drop=True)    
        df.to_csv(name+categories[i]+'.csv', index=True, index_label='Index')
        
def data_save_iterator(N=None, n=None, Prefix=None):
    
    if(N==None):
        N=int(input('Enter number of files to produce (N):'))
    if(n==None):
        n=int(input('Enter number of samples in each file (n):'))
    if(Prefix==None):
        Prefix=input('Enter prefix for files produced by the program:')
    for i in range(N):
        t0 = time.time()
        data_saver('dataJK/'+Prefix+str(i), n)
        deltat = time.time() - t0 
        print(f'File {i+1} of {N} saved. Total time: {deltat:.2f}')

class samples():
    def __init__(self, df=None):
        if(df==None):
            self.Bins = pd.DataFrame()
            self.Matrix = pd.DataFrame()
            self.OptimalState = pd.DataFrame()
            self.Measures = pd.DataFrame()
            self.Rotation = pd.DataFrame()
        else:
            self.Bins = df.Bins
            self.Matrix = df.Matrix
            self.OptimalState = df.OptimalState
            self.Measures = df.Measures
            self.Rotation = df.Rotation
    
    def save(self, destination):
        self.Bins.to_csv(destination+'Bins.csv', index_label='Index')
        self.Matrix.to_csv(destination+'Matrix.csv', index_label='Index')
        self.OptimalState.to_csv(destination+'OptimalState.csv', index_label='Index')
        self.Measures.to_csv(destination+'Measures.csv', index_label='Index')
        self.Rotation.to_csv(destination+'Rotation.csv', index_label='Index')
    
    def read(self, destination):
        self.Bins = pd.read_csv(destination+'Bins.csv', index_col='Index')
        self.Matrix = pd.read_csv(destination+'Matrix.csv',index_col='Index')
        self.OptimalState = pd.read_csv(destination+'OptimalState.csv', index_col='Index')
        self.Measures = pd.read_csv(destination+'Measures.csv', index_col='Index')
        self.Rotation = pd.read_csv(destination+'Rotation.csv', index_col='Index')
        for col in self.Matrix.columns:
            self.Matrix[col] = self.Matrix[col].apply(complex)
    
    def histogram(self, index):
        if(type(index)==int):
            plt.stairs(self.Bins.iloc[index].values)
        else:
            raise ValueError('Index must be an int')

    def opt_histogram(self, index):
        if(type(index)==int):
            density_matrix(rho2(self.OptimalState.loc[index].Angle, self.OptimalState.loc[index].Visibility)).histogram()            
        else:
            raise ValueError('Index must be an int')
        
    def double_plot(self, index):
        if(type(index)==int):
            double_plot(density_matrix(rho2(self.OptimalState.iloc[index].Angle, self.OptimalState.iloc[index].Visibility)),
                self.Bins.iloc[index].values)
        else:
            raise ValueError('Index must be an int')

    def density_matrix(self, idx):
        return density_matrix(np.reshape(self.Matrix.iloc[idx], (4,4)))
    
    def opt_density_matrix(self, idx):
        return density_matrix(rho2(self.OptimalState.Angle.iloc[idx], self.OptimalState.Visiblity.iloc[idx]))
       
        
def load_samples(destination, categories = ['Matrix', 'Measures', 'Bins', 'OptimalState', 'Rotation']):
    samps = samples()
    names = ['Matrix', 'Measures', 'Bins', 'OptimalState', 'Rotation']
    for cat in categories:
        if cat in names:
            exec('samps.'+ cat +' = pd.read_csv(destination+cat+".csv", index_col="Index")')
    if 'Matrix' in categories:
        for col in samps.Matrix.columns:
            samps.Matrix[col] = samps.Matrix[col].apply(complex)
    return samps



def join_data(alldata):
    from os import listdir
    names = ['Matrix', 'Measures', 'Bins', 'OptimalState', 'Rotation']
    all=[eval('alldata.'+name).loc[[False]*len(alldata.Matrix)] for name in names]
    for f in listdir('dataJK/obliczenia_2'):
        for i, name in enumerate(names):
            if name in f:
                all[i] = pd.concat((all[i], pd.read_csv('dataJK/obliczenia_2/'+f, index_col='Index')))
    
    for i, name in enumerate(names):
        all[i].to_csv('all_complex'+name+'.csv')             
        
        