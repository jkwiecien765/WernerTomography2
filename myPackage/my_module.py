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


def unitary_mat2(params):
    
    th = params[0]
    alpha = params[1]
    beta = params[2]
    u1=np.array([[np.exp(1j* alpha)*np.cos(th), np.exp(1j* beta)*np.sin(th)],\
                    [-np.exp(-1j* beta)*np.sin(th),np.exp(-1j* alpha)*np.cos(th)]])
    return u1

def unitary_mat3(L=2):
    #mat=np.zeros((4,4))
    
    mat=np.random.normal(size=(L,L,2)).view(np.complex128).reshape(L,L,)
    mat2=mat.copy()
    for i in range(L):
            for j in range(i):
                mat2[i] -= mat2[j] * np.inner(np.conjugate(mat2[j]), mat[i]) / np.inner(np.conjugate(mat2[j]), mat2[j])
            print(mat2[i], end=' ')
            mat2[i] /= np.sqrt(np.real(np.inner(np.conjugate(mat2[i]), mat2[i])))
            print(mat2[i])
    return np.array(mat2)

#Density matrix of Werner state and its generalisation
def rho2(th, vis):
    '''returns 4x4 ndarray matrix of Werner state
    first argument is angle of sine in the formula, second is the visibility'''
     
    entgl=np.outer(np.sin(th),ZeroOne).flatten()+np.outer(np.cos(th),OneZero).flatten()
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



def rotate_matrix(matrix, paramsA, paramsB):
    matrix = matrix.matrix if type(matrix) == density_matrix else matrix
    uA = unitary_mat2(paramsA)
    uB = unitary_mat2(paramsB)
    uAB = tens_prod2d(uA, uB)
    return np.transpose(np.conjugate(uAB))@matrix@uAB   

def obs(rho,parA = None, parB = None):
    '''Simulation of observation of density matrix with unitary matrices of given parameters (defaults to random) 
        returns probability of observation as being in 00 state'''
    parA = rand_phase() if parA == None else parA
    parB = rand_phase() if parB == None else parB
    uA = unitary_mat2(parA)
    uB = unitary_mat2(parB)    
    u=tens_prod2d(uA,uB)
    zer=np.outer(ZeroZero,ZeroZero)
    p=rho@(np.transpose(np.conjugate(u)))@zer@u
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

def double_plot(dmA, dmB, show_fidelity=False):
    binsA = dmA.bins()
    binsB = dmB.bins()
    plt.stairs(binsA['counts'], binsA['bins'], fill=True)
    plt.stairs(binsB['counts'], binsB['bins'])
    if show_fidelity:
        mins = classical_fidelity2(dmA, dmB)[1]
        plt.stairs(mins, binsB['bins'], color = 'black')

def matrix_fidelity(matrixA, matrixB):
    '''Calculates matrix fidelity given two ndarrays of matrices (or density_matrices)'''

    if(type(matrixA) == density_matrix):
        matrixA = matrixA.matrix
    if(type(matrixB) == density_matrix):
        matrixB = matrixB.matrix
    
    fid = min(np.real(np.trace(sqrtm(sqrtm(matrixA)@matrixB@sqrtm(matrixA))))**2,1)
    
    return fid

def optimal_matrix_fidelity(dmA):
    from scipy.optimize import differential_evolution
    def f(params, matrixA):
        matrixA = matrixA.matrix if type(matrixA) == density_matrix else matrixA
        matrixB = rho2(params[-1], 1)
        paramsA = params[:3]
        paramsB = params[3:-1]
        return -1*matrix_fidelity(rotate_matrix(matrixB, paramsA, paramsB), matrixA)
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
    
    
#Optimises visibility to match a state (given in bin counts)
   
def vis_optimizer(binsIn, matrix, plot=False, frob=False):
    '''Optimises a state's (2nd argument) visibility with respect to experimental data in bins (or density_matrix or simple 4x4 ndarray)'''
    
    if(frob and type(binsIn)==np.ndarray):
        if(binsIn.shape!=(4,4)):
            raise TypeError('If you want Frobenius distance input must be 4x4 ndarray or density_matrix')
            
    if(type(binsIn) == density_matrix):
        matrixIn=binsIn.matrix
        binsIn = binsIn.bins()['counts']
    elif(np.shape(binsIn) == (4,4)):
        matrixIn=binsIn
        binsIn = density_matrix(binsIn).bins()['counts']
            
    fid = classical_fidelity(binsIn, matrix.bins()['counts'])
    if frob:
        dist=Frobenius_dist(matrixIn, matrix)
        print(f'Initial distance {dist}')
        
    print(f'Initial fidelity: {fid}')
    if(fid < 0.25):
        vis = 0
    else:
        vis = 4/3 * (fid - 1/4)
    print(f'Optimal visibility: {vis}')
    opt_matrix=matrix*vis+(1-vis)*density_matrix(np.diag([0.25,0.25,0.25,0.25]))
    
    if frob:
        fid=classical_fidelity(binsIn, opt_matrix.bins()['counts'])
        qf=matrix_fidelity(matrixIn, opt_matrix)
        print(f'Final classical fidelity: {fid}')
        print(f'Quantum fidelity: {qf}')
        dist=Frobenius_dist(matrixIn, opt_matrix)
        print(f'Final distance: {dist}')
        th_dist=np.sqrt(1 - 4/3 * qf * qf + 2/3 * qf - 1/3)
        print(f'Theoretically predicted distance: {th_dist}')
        
    if(plot):
        plt.stairs(binsIn, np.linspace(0, 1, len(binsIn)+1), fill=True)
        plt.hist(opt_matrix.data,bins=len(binsIn), density=True)
        plt.show
    
    return opt_matrix


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
        vis = 4/3 * (mf - 1/4)
    if printing:
        print(f'Optimal visibility: {vis}')
    opt_matrix=dm1*vis+(1-vis)*density_matrix(np.diag([0.25,0.25,0.25,0.25]))

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


def rand_PSDM():
    '''Generates a 4x4 matrix of a Positive Semi-definite matrix with trace 1'''
    mat=np.array(np.random.rand(4,4))
    # Any matrix that is product of B.BT where B is a real-valued invertible matriix is PSDM 
    PSDM = np.matmul(mat,np.transpose(mat))
    PSDM /= np.trace(PSDM)
    
    if(abs(1-np.trace(PSDM))>1e-7):
        print(np.trace(PSDM))
        raise Exception('Fail: tr!=1')
    
    return PSDM
        
def mean_over_unitars(matrix, N=100000, recording=False):
    '''Takes a matrix or 4x4 list/ndarray and translates it N times over unitary matrices. If recording=True, it returns also a pandas.DataFrame with each iteration of the loop'''
    matrix=np.array(matrix)
    record=pd.DataFrame()
    for param in parameters[0:N]:
        if recording:
            ser=pd.Series(np.append(np.asarray(matrix).flatten(), np.trace(matrix))).to_frame().T
            record=pd.concat([record, ser])    
        uA=np.array(unitary_mat2(param[0]))
        uB=np.array(unitary_mat2(param[1]))
        u=tens_prod2d(uA,uB)
        matrix = u@matrix@(np.transpose(np.conjugate(u)))
        matrix = np.real(matrix)
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
    final_matrix = np.array(np.zeros([4,4]))
    record = pd.DataFrame()
    matrix=final_matrix
    ser=pd.Series(np.asarray(matrix).flatten()).to_frame().T
    record=pd.concat([record, ser])
    
    for param in parameters[0:N]:
        
                
        uA=np.array(unitary_mat2(param[0]))
        uB=np.array(unitary_mat2(param[1]))
        u=tens_prod2d(uA,uB)
        final_matrix += np.real(u@initial_matrix@(np.transpose(np.conjugate(u)))/N)
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
    rhod = tens_prod2d(Pauli[2], Pauli[2])@np.transpose(np.conjugate(rho))@tens_prod2d(Pauli[2], Pauli[2])
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
        if isinstance(num, float):
            return(density_matrix(num*self.matrix))
        else:
            raise TypeError("You must multiply by a float")

    def __rmul__(self, num):
        if isinstance(num, float):
            return(density_matrix(num*self.matrix))
        else:
            raise TypeError("You must multiply by a float")
    
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
            print("setting density_matrix data...")
            self.set() 
        mi=np.round(min(self.data),3)
        ma=np.round(max(self.data),3)
        return(mi, ma)
    
    def histogram(self, BinNum=100, AdjustBins=False):
        if len(self.data)==0:    
            print("setting density_matrix data...")
            self.set()
            
        if(AdjustBins):
            ran=max(self.range()[1]-self.range()[0],0.001)
            plt.hist(self.data,int(BinNum/ran),range=(0,1),density=True)
        else:
            plt.hist(self.data,BinNum,range=(0,1),density=True)
    
    def bins(self, BinNum=100, AdjustBins=False):
        if len(self.data)==0:    
            print("setting density_matrix data...")
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
    ans = optimal_matrix_fidelity(dm)
    angle = ans['angle']
    rotation = ans['parameters']
    opt_matrix, vis = vis_optimizer_dm(dm, density_matrix(rotate_matrix(rho2(angle, 1), rotation[0], rotation[1])), printing = False)
    opt_matrix.name = 'angle=' + str(angle) + ', vis=' + str(vis)
    opt_matrix.matrix = opt_matrix.matrix/np.trace(opt_matrix.matrix)
    hist = dm.bins()['counts'].tolist()
    
    return {'Matrix': dm.matrix.tolist(), 'Bins': hist, 'Angle': angle, 'Visibility': vis, 'Rotation': rotation,
            'Distance': Frobenius_dist(dm, opt_matrix), 'MatrixFidelity': matrix_fidelity(dm, opt_matrix),
            'HistogramFidelity': classical_fidelity(dm, opt_matrix), 'Covering': classical_fidelity2(dm, opt_matrix),
            'ConcurrenceOriginal': concurrence(dm), 'ConcurrenceOpt': concurrence(opt_matrix), 
            'CHSHViolationMOriginal': CHSHviolation_measure(dm), 'CHSHViolationMOpt': CHSHviolation_measure(opt_matrix)}

def data_order(dictionary):
    binsDF = pd.DataFrame(dictionary['Bins'])
    bins = np.linspace(0,1,101)
    bins2 = []
    for i in range(100):
        bins2.append('[' + str(round(bins[i],2)) + ', ' + str(round(bins[i+1],2)) + ']')
    bins2 = pd.Series(bins2, name='Index')
    bins3 = ['Bins']*100
    bins3 = pd.Series(bins3, name='Category')
    binsDF = pd.merge(binsDF, bins2, left_index=True, right_index=True)
    binsDF = pd.merge(binsDF, bins3, left_index=True, right_index=True)
    binsDF.set_index(['Category','Index'], inplace=True)
    binsDF = binsDF.transpose()
    matrixList = []
    matrixIndex = []
    matrixType = ['Matrix'] * 16
    for i in range(4):
        for j in range(4):
            matrixList.append(dictionary['Matrix'][i][j])
            matrixIndex.append(str(i)+','+str(j))
    matrixDF = pd.DataFrame({'Index': matrixIndex, 0: matrixList, 'Category': matrixType})
    matrixDF.set_index(['Category', 'Index'], inplace=True)
    matrixDF = matrixDF.transpose()
    allDF=pd.merge(binsDF, matrixDF, left_index=True, right_index=True)
    
    rotationList = []
    rotationIndexl0 = []
    rotationIndexl1 = ['Rotation']*6
    for i in range(2):
        for j in range(3):
            rotationList.append(dictionary['Rotation'][i][j])
            rotationIndexl0.append(3*i+j)
    rotationDF = pd.DataFrame({'Category': rotationIndexl1, 'Index': rotationIndexl0, 0: rotationList})
    rotationDF = rotationDF.set_index(['Category', 'Index']).transpose()
    allDF = pd.merge(allDF, rotationDF, left_index=True, right_index=True)
    
    paramsList = [dictionary['Angle'], dictionary['Visibility']]
    paramsIndexl0= ['Angle', 'Visibility']
    paramsIndexl1 = ['OptimalState']*2
    paramsDF = pd.DataFrame({'Category': paramsIndexl1, 'Index': paramsIndexl0, 0: paramsList})
    paramsDF = paramsDF.set_index(['Category', 'Index']).transpose()
    allDF = pd.merge(allDF, paramsDF, left_index=True, right_index=True)
    
    measuresIndexl1 = ['Measures'] * 8
    measuresIndexl0 = ['Distance',  'MatrixFidelity', 'HistogramFidelity', 'Covering', 'ConcurrenceOriginal', 'ConcurrenceOpt', 'CHSHViolationMOriginal', 'CHSHViolationMOpt']
    measuresList = [dictionary[key] for key in measuresIndexl0]
    measuresDF = pd.DataFrame({'Category': measuresIndexl1, 'Index': measuresIndexl0, 0: measuresList})
    measuresDF = measuresDF.set_index(['Category', 'Index']).transpose()
    allDF = pd.merge(allDF, measuresDF, left_index=True, right_index=True)    
        
    return allDF


def data_saver(name, N=1000):
    df = data_order(data_generator())
    for i in range(N-1):
        df=pd.concat((df, data_order(data_generator())))
    
    df = df.reset_index().drop('index', axis=1)    
    df.transpose().to_csv(name, index=True)
    df = pd.read_csv(name)
    df = df.set_index(['Category', 'Index']).transpose()
    
def data_save_iterator(N=1000, n=1000, Prefix=''):
    for i in range(N):
        data_saver('dataJK/'+Prefix+'data'+str(i)+'.csv', n)

def data_reader(directory='dataJK'):
    import os
    df = pd.DataFrame()
    for file in os.listdir(directory):
        temp = pd.read_csv(directory+'/'+file, index_col=['Category', 'Index']).transpose()
        df = pd.concat((df,temp))
    return df.reset_index().drop('index', axis=1)