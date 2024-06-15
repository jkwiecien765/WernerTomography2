# %% Inital denfinitions and imports

# Functions for Werner app
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections.abc import Iterable
from scipy.linalg import sqrtm
from numpy import pi
parameters=np.load('parameters.npy')
## Useful matrices

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

## Basic mathematical operations

#Tensor product which doesn't contain nested list in output
def tens_prod2d(u1,u2):

    U=np.tensordot(u1,u2,0)
    ua=np.concatenate((U[0][0],U[0][1]),1)
    ub=np.concatenate((U[1][0],U[1][1]),1)
    u3=np.concatenate((ua,ub),0)
    return np.array(u3)


#3 random numbers [0, 2Pi] with Haar measure on SU(2)

def rand_phase():
    r=[np.arcsin(np.sqrt(np.random.rand())),np.random.rand()*2*pi,np.random.rand()*2*pi]
    return r


# Unitary matrices with given parameters or random parameters

def unitary_mat_par(params):
    th = params[0]
    alpha = params[1]
    beta = params[2]
    u1=np.array([[np.exp(1j* alpha)*np.cos(th), np.exp(1j* beta)*np.sin(th)],\
                    [-np.exp(-1j* beta)*np.sin(th),np.exp(-1j* alpha)*np.cos(th)]])
    return u1

def unitary_mat_rand(L=2):
    from scipy import stats
    mat = stats.unitary_group.rvs(L)
    mat = mat / np.power(np.linalg.det(mat), 1/L)
    return np.array(mat)


def rotate_matrix_par(matrix, paramsA, paramsB):
    matrix = matrix.matrix if type(matrix) == density_matrix else matrix
    uA = unitary_mat_par(paramsA)
    uB = unitary_mat_par(paramsB)
    uAB = tens_prod2d(uA, uB)
    return np.transpose(np.conjugate(uAB))@matrix@uAB 


def rotate_matrix(matrix):
    matrix = matrix.matrix if type(matrix) == density_matrix else matrix
    uA = unitary_mat_rand(2)
    uB = unitary_mat_rand(2)
    uAB = tens_prod2d(uA, uB)
    return np.transpose(np.conjugate(uAB))@matrix@uAB   

#Density matrix of Werner state and its generalisation
def rho2(th, vis):
    '''returns 4x4 ndarray matrix of Werner state
    first argument is angle of sine in the formula, second is the visibility'''
     
    entgl=np.sin(th)*ZeroOne+np.cos(th)*OneZero
    return vis * np.outer(entgl,entgl)\
          + (1-vis)/4 * np.identity(4)



#Random density matrix - copied from qiskit
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

def obs(rho,parA = None, parB = None):
    '''Simulation of observation of density matrix with unitary matrices of given parameters (defaults to random) 
        returns probability of observation as being in 00 state'''
    
    uA = unitary_mat_rand() if parA==None else unitary_mat_par(parA)
    uB = unitary_mat_rand() if parB==None else unitary_mat_par(parB)
    u = tens_prod2d(uA,uB)
    zer = np.outer(ZeroZero,ZeroZero)
    p = rho@(np.transpose(np.conjugate(u)))@zer@u
    return np.real(np.trace(p))


## Class that represents quantum states (density matrix). It takes a 4x4 matrix for init    
class density_matrix:
    
    def __init__(self, rho, name=''):
        if np.shape(rho)!=(4,4):
            raise TypeError("Density matrix must be 4x4 array")
        self.matrix=np.array(rho)
        self.name=name
    
    #Can be added to other density matrices        
    def __add__(self, density_matrix2):
        return(density_matrix(self.matrix+density_matrix2.matrix))
    
    #Multiplied by a float
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
    
    data=[]
    
    #Runs random measurement N times and stores the result
    def set(self,N=50000, start=0):
        self.data=[]
        params=parameters[start:start+N]
        for i in range(len(params)):
            self.data.append(obs(self.matrix,params[i][0],params[i][1]))    
        self.data = np.array(self.data)
    
    #Produces a histogram of probabilities    
    def histogram(self, BinNum=100):
        if len(self.data)==0:    
            self.set()
        plt.hist(self.data,BinNum,range=(0,1),density=True)
    
    #Produces bins of probabilities
    def bins(self, BinNum=100):
        if len(self.data)==0:    
            self.set() 
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



#Produces two histograms of probability in one figure
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


## Quantum properties
    
def fidelity(matrixA, matrixB):
    '''Calculates fidelity given two ndarrays of matrices (or density_matrices)'''

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

def Frobenius_dist(A, B):
    '''Frobenius distance of two states. Input must me two 4x4 matrices or density_matrices'''
    A=A.matrix if type(A)==density_matrix else A
    B=B.matrix if type(B)==density_matrix else B
        
    D=A-B
    dist=np.sqrt(np.real(np.trace(np.transpose(np.conjugate(D))@D)))
    return dist

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

####### FITTING THE CLOSEST WERNER STATE ########
#%%
## Optimizer for fitting the closest Werner State

def optimal_matrix_fidelity(dmA):
    from scipy.optimize import differential_evolution
    def f(params, matrixA):
        matrixA = matrixA.matrix if type(matrixA) == density_matrix else matrixA
        matrixB = rho2(params[-1], 1)
        paramsA = params[:3]
        paramsB = params[3:-1]
        
        return -1*fidelity(matrixA,rotate_matrix(matrixB, paramsA, paramsB))
    bounds = [(0,2*pi), (0,2*pi), (0,2*pi), (0,2*pi), (0,2*pi), (0,2*pi), (0, pi/4)]
    res = differential_evolution(f, args=(dmA,), bounds=bounds)
    return {'value': -res['fun'], 'angle': res['x'][-1], 'parameters': [res['x'][:3].tolist(), res['x'][3:6].tolist()]}

def vis_optimizer_dm(dm2, dm1):
    '''Optimises a state's (2nd argument) visibility with respect to experimental data in bins (or density_matrix or simple 4x4 ndarray)'''
    mf=fidelity(dm2, dm1)
    dist=Frobenius_dist(dm2, dm1)
    if(mf < 0.25):
        vis = 0.0
    else:
        vis = 1.3333333333333333333 * (mf - 0.25)
        
    opt_matrix=dm1*vis+(1.0-vis)*density_matrix(np.diag([0.25,0.25,0.25,0.25]))
    return opt_matrix, vis


#%% Data generation, ordering & saving (definitions)
import time

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
    opt_matrix.matrix = opt_matrix.matrix/np.trace(opt_matrix.matrix)
    hist = np.real(dm.bins()['counts']).tolist()
    
    return {'Matrix': dm.matrix.tolist(), 'Bins': hist, 'Angle': angle, 'Visibility': np.real(vis), 'Rotation': rotation,
            'Distance': np.real(Frobenius_dist(dm, opt_matrix)), 'MatrixFidelity': np.real(fidelity(dm, opt_matrix)),
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
    
    measuresIndexl0 = ['Distance',  'MatrixFidelity', 'ConcurrenceOriginal', 'ConcurrenceOpt', 'CHSHViolationMOriginal', 'CHSHViolationMOpt']
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


'''Class for storing data'''

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

#%% Data generation (run)
data_save_iterator(N=10, n=10, Prefix='example')
        
####### END OF FITTING THE CLOSEST WERNER STATE ########

####### PROJECTION ONTO SYMMETRIC STATES ########


#%% Basic definitions 

def wer(th, v=1, base='psim'):
    '''
    Creates Werner state based on parameters of angle, visibility and Bell state
    Args:
        th: angle
        v: visibility (default=1)
        base: which Bell state to choose. Accepts one of the strings: "psip", "psim", "phip", "phim"
        which means phi/psi with plus (p) or minus (m) sign (default="psim")
    Returns:
        4x4 np.ndarray
    
    '''
    match base:
        case 'psim':
            vec = np.sin(th)*OneZero - np.cos(th)*ZeroOne
        case 'psip':
            vec = np.sin(th)*OneZero + np.cos(th)*ZeroOne
        case 'phip':
            vec = np.sin(th)*OneOne + np.cos(th)*ZeroZero
        case 'phim':
            vec =np.sin(th)*OneOne - np.cos(th)*ZeroZero
        case _:
            raise ValueError("You must choose one of the following: psip, psim, phip, phim")
    return v * np.outer(vec,vec) + (1-v)/4 * np.identity(4)


# Projection using the explicit formula
def projection(matrix):
        return np.real(np.matrix([[matrix[0,0],0,0,matrix[0,3]/2+matrix[3,0]/2],\
                                  [0,matrix[2,2]/2+matrix[1,1]/2,0,0],\
                                  [0,0,matrix[2,2]/2+matrix[1,1]/2,0],\
                                  [matrix[0,3]/2+matrix[3,0]/2,0,0,matrix[3,3]]]))

# Finding optimal rotation and projecting
def phi_proj(matrixA, target='fidelity'):
    '''
    Performs a unitary rotation and projects onto phi+ state (optimizes for fidelity, distance or concurrence)
    Args:
        matrixA: matrix to be projected
        target: function to be optimizes for. One of the following: 'fidelity', 'distance' or 'concurrence'
    Returns:
        A dict object:
            fidelity/distance/concurrence: optimal value of the target function
            parameters: 2x3 angles of unitary rotation
            angle: angle of resulting phi+ state
            visiblity: visibility of resulting phi+ state
            projected: np.matrix result of projection
            original: np.matrix original matrix
    '''
    def get_params_phi(matrix):
        v = 1 - 4*matrix[1,1]
        if v==0:
            return 0, 0    
        alph = np.arcsin(matrix[0,3]/v*2)/2
        return alph, v    
    from scipy.optimize import differential_evolution
        
    def f(params, matrixA):
        matrixA = matrixA.matrix if type(matrixA) == density_matrix else matrixA
        paramsA = params[:3]
        paramsB = params[3:]
        projected = projection(matrixA)
        rotated = rotate_matrix_par(projected, paramsA, paramsB)
        if target=='fidelity':
            return -1*fidelity(density_matrix(rotated), matrixA)
        if target=='concurrence':
            return -1*concurrence(rotated)
        if target=='distance':
            return Frobenius_dist(matrixA, rotated)
    bounds = [(0,2*pi), (0,2*pi), (0,2*pi), (0,2*pi), (0,2*pi), (0,2*pi)]
    res = differential_evolution(f, args=(matrixA,), bounds=bounds, workers=1)
    
    if target=='distance':
        res['fun'] = -res['fun']
        
    paramsA = res['x'][:3]
    paramsB = res['x'][3:]
    matrix_proj =rotate_matrix_par(projection(matrixA), paramsA, paramsB)
    angle, vis = get_params_phi(matrix_proj)
    dist = Frobenius_dist(matrix_proj, matrixA)
    conc = concurrence(density_matrix(matrix_proj))
    conc_org = concurrence(density_matrix(matrixA))
    fid = fidelity(matrixA, matrix_proj)   
    return {'distance': dist, 'concurrence' : conc, 'fidelity': fid,\
            'parameters': [paramsA, paramsB], 'angle': angle,\
            'visibility': vis, 'projected': matrix_proj, 'original': matrixA, 'target': target, 'concurrence_org':conc_org}    

# Better matrix printig
def matrix_print(func):
    def wrapper(*args, **kwargs):
        out=func(*args, **kwargs)
        out = np.array(out) if type(out)=='numpy.matrix' else out
        s = [[str(f'{e.real:.3f} + {e.imag:.3f}i ') for e in row] for row in out]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print('\n'.join(table))
    return wrapper

# Projection by integration

@matrix_print
def int_projection(dm, N=10000):

    qubit_switch = np.matrix([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
        
    def phase_rotation(phi=np.random.random()*2*pi):
        from scipy.linalg import expm
        sig_z = np.array([[1,0],[0,-1]], dtype='complex')
        U = tens_prod2d(expm((1j * phi * sig_z)), expm((-1j * phi * sig_z)))
        return U
    
    def phase_rotation_dag(phi=np.random.random()*2*pi):
        return np.transpose(np.conjugate(phase_rotation(phi)))
    
    proj = np.zeros(shape=(4,4), dtype='complex')
    dm = dm.matrix if type(dm)=='density_matrix' else dm
    for i in range(N):
        phi=np.random.random()*2*pi
        rotated = phase_rotation(phi)@dm@phase_rotation_dag(phi)
        switched = qubit_switch@rotated@qubit_switch
        proj += (np.transpose(rotated + switched) + rotated + switched)/4
    
    proj = proj/N
    return proj
            
# Data generation & save (definitions)

df_raw=pd.DataFrame({'distance':[], 'concurrence':[],'fidelity':[], 'target':[], 'concurrence_org':[]})
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    for i in range(10000):
        state = rand_PSDM(4)
        res = phi_proj(state, target='distance')
        df_raw.loc[len(df_raw)] = [res['distance'], res['concurrence'], res['fidelity'], res['target'], res['concurrence_org']]
        res = phi_proj(state, target='fidelity')
        df_raw.loc[len(df_raw)] = [res['distance'], res['concurrence'], res['fidelity'], res['target'], res['concurrence_org']]
        res = phi_proj(state, target='concurrence')
        df_raw.loc[len(df_raw)] = [res['distance'], res['concurrence'], res['fidelity'], res['target'], res['concurrence_org']]
        if i%10 == 9:
            print(f'{i+1} out of 10000 done')        
    
df_raw.to_csv('projection.csv', index=False)

# %% Visualisation of integral projection converging to the result of explicit formulas (definitions)

def int_projection_conv(dm, N=10000):

    qubit_switch = np.matrix([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
        
    def phase_rotation(phi=np.random.random()*2*pi):
        from scipy.linalg import expm
        sig_z = np.array([[1,0],[0,-1]], dtype='complex')
        U = tens_prod2d(expm((1j * phi * sig_z)), expm((-1j * phi * sig_z)))
        return U
    
    def phase_rotation_dag(phi=np.random.random()*2*pi):
        return np.transpose(np.conjugate(phase_rotation(phi)))
    
    def projection(matrix):
        return np.real(np.matrix([[matrix[0,0],0,0,matrix[0,3]/2+matrix[3,0]/2],\
                                    [0,matrix[2,2]/2+matrix[1,1]/2,0,0],\
                                    [0,0,matrix[2,2]/2+matrix[1,1]/2,0],\
                                    [matrix[0,3]/2+matrix[3,0]/2,0,0,matrix[3,3]]]))        
    
    proj = np.zeros(shape=(4,4), dtype='complex')
    errs = np.zeros(shape = N)
    dm = dm.matrix if type(dm)=='density_matrix' else dm
    for i in range(N):
        phi=np.random.random()*2*pi
        rotated = phase_rotation(phi)@dm@phase_rotation_dag(phi)
        switched = qubit_switch@rotated@qubit_switch
        proj += (np.transpose(rotated + switched) + rotated + switched)/4
        errs[i] = Frobenius_dist(projection(dm), proj/(i+1))
    
    proj = proj/N
    return errs

# %% Visualisation of integral projection converging to the result of explicit formulas (run)

hist=[]
for i in range(5):
    hist.append(int_projection_conv(rand_PSDM(4), N=1000000))   

plt.yscale('log')
plt.xscale('log')
plt.grid(True)
plt.xlabel('Number of iterations')
plt.ylabel('Frobenius distance')
plt.title('Convergence of integral projection')
for i in range(3):
    plt.plot(hist[-i-1])

plt.show()

# %% Data analysis
df_raw = pd.read_csv('projection.csv')
df = df_raw.copy(deep=True)
df['concurrence_diff'] = df.concurrence_org - df.concurrence
df.loc[df['concurrence_org']==0, 'concurrence_diff']=np.nan
df['1 - fidelity'] = 1 - df['fidelity']
import seaborn as sns
df_melt=df.melt(['target'], value_vars = ['1 - fidelity', 'distance', 'concurrence_diff'], var_name='quantity')
df_melt['target'] = df_melt['target'].astype('category')
plt.ylim(-0.1,1)
sns.barplot(data=df_melt,  orient='v', x='target', y='value', hue='quantity', errorbar='sd', estimator='median')
plt.show()

#%% Data analysis continued
df_melt.loc[np.logical_and(df_melt.target=='concurrence', df_melt.quantity=='concurrence_diff')].hist(bins = np.arange(0, 0.6, 0.025))
plt.title('Concurrence difference while optimising for maximum concurrence')
plt.xlabel('Concurrence difference')
plt.ylabel('Counts')
plt.grid(False)
plt.show()


####### END OF PROJECTION ONTO SYMMETRIC STATES ########

####### NEURAL NETWORKS ########
# %% Necessary imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

####### CASE OF GENERIC STATES #######
# %% Data load, reformatting & save
data = load_samples('all_complex')
samps = [(x, y) for x, y in zip(data.Bins.values, data.OptimalState.values)]
import pickle
with open('ML_all_samples.dat', 'wb') as f:
    pickle.dump(samps, f)

# %% Reformatted data load    
import pickle
with open('ML_all_samples.dat', 'rb') as f:
    samps = pickle.load(f)
    
# %% Definition of dataset and neural network architecture
class all_dataset(Dataset):
    
    def __init__(self,samples):
        super(Dataset, self).__init__()
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        hist, params = self.samples[idx]
        hist = torch.from_numpy(hist).float()
        params = torch.Tensor(params).float()
        return hist, params
    
class Net(nn.Module):
    
    def __init__(self, input=100, output=2):
        super().__init__()
        
        self.input_layer = nn.Linear(input, 50)
        self.output_layer = nn.Linear(20, output)
        self.hidden_layers = nn.Sequential(
            nn.Linear(50, 100),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(100,50),
            nn.Sigmoid(),
            nn.Dropout(p=0.3),
            nn.Linear(50,70),
            nn.Sigmoid(),
            nn.Dropout(p=0.3),
            nn.Linear(70,20),
            nn.Sigmoid(),
            nn.Dropout(p=0.3),
        )
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x   
        
#%%
#Trainig & validation loop
def fit(model, samples_train, samples_val, batch_size=10, lr=0.05, epochs=10, dataset = all_dataset, criterion=nn.MSELoss()):    
    from torchmetrics import MeanSquaredError
    data_loader_train = DataLoader(dataset(samples_train), shuffle=True, batch_size=batch_size)
    data_loader_val = DataLoader(dataset(samples_val), shuffle=True, batch_size=batch_size)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=0.5) 
    metrics = MeanSquaredError()

    for epoch in range(epochs):
        #Training
        model.train()
        for hist, params in data_loader_train:
            optimizer.zero_grad()
            pred = model(hist)
            loss = criterion(pred, params)
            loss.backward()
            optimizer.step()
            metrics(pred, params)
        train_loss = metrics.compute()
        metrics.reset()
        #Validation
        model.eval()
        with torch.no_grad():
            for hist, params in data_loader_val:
                pred = model(hist)
                metrics(pred, params)
        val_loss = metrics.compute()
        metrics.reset()
        
        print(f'Epoch {epoch+1}/{epochs}: training loss: {train_loss:.5f}, validation loss: {val_loss:.5f}')
    
    return model    
    
# %%
# Evaluation
def evaluate_model(model, samples_eval, batch_size=10, dataset=all_dataset):
    from torchmetrics import MeanSquaredError
    metrics = MeanSquaredError()
    data_loader = DataLoader(dataset(samples_eval), shuffle=True, batch_size=batch_size)
    model.eval()
    with torch.no_grad():
        for hist, params in data_loader:
            pred = model(hist)
            metrics(pred, params)
    return metrics.compute()

#%% Training
model = fit(Net(), samps[:200000], samps[200000:250000], epochs=10, batch_size=100)

#%% Comparison of model predicion for two inputs
model.eval()
print(model(torch.Tensor(samps[15][0]).float()))
print(model(torch.Tensor(samps[10][0]).float()))
####### END OF CASE OF GENERIC STATES #######

####### CASE OF PROJECTED STATES #######
# %% Integral projection data generation (definitions)
def data_to_bins(data):
    counts=np.zeros(100)
    for dat in data:
        try:
            counts[int(dat*100)]+=1/len(data)
        except IndexError:
            pass
    return counts

def projection(matrix):
    return np.real(np.matrix([[matrix[0,0],0,0,matrix[0,3]/2+matrix[3,0]/2],\
                                [0,matrix[2,2]/2+matrix[1,1]/2,0,0],\
                                [0,0,matrix[2,2]/2+matrix[1,1]/2,0],\
                                [matrix[0,3]/2+matrix[3,0]/2,0,0,matrix[3,3]]]))
qubit_switch = np.matrix([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
    
def phase_rotation(phi=np.random.random()*2*pi):
    from scipy.linalg import expm
    sig_z = np.array([[1,0],[0,-1]], dtype='complex')
    U = tens_prod2d(expm((1j * phi * sig_z)), expm((-1j * phi * sig_z)))
    return U

def phase_rotation_dag(phi=np.random.random()*2*pi):
    return np.transpose(np.conjugate(phase_rotation(phi)))

#%% Integral projection data generation (run) & save
proj_bins=[]
N=10000
for i in range(N):
    initial_matrix = rand_PSDM(4)
    projected = projection(initial_matrix)
    parameters = [projected[0,0], projected[0,3], projected[1,1], projected[3,3]]
    proj_data=[]
    for j in range(2000):
        phi=np.random.random()*2*pi
        rotated = phase_rotation(phi)@initial_matrix@phase_rotation_dag(phi)
        switched = qubit_switch@rotated@qubit_switch
        proj_data.append(np.trace(np.real(switched@np.outer(ZeroZero,ZeroZero))))
    
    proj_bins.append(data_to_bins(proj_data))
    
    if (i+1)%10 == 0:
        print(f'{i+1}/{N} done')
        
import pickle
if len(proj_bins) > 10:
    with open('ML_int_proj.dat', 'wb') as file:
        pickle.dump(proj_bins, file)

# %% Integral projection data load
import pickle
with open('ML_int_proj.dat', 'rb') as file:
    proj_bins_load = pickle.load(file)
    
# %% Explicit data projection (run) & save
def projection(matrix):
    return np.real(np.matrix([[matrix[0,0],0,0,matrix[0,3]/2+matrix[3,0]/2],\
                                [0,matrix[2,2]/2+matrix[1,1]/2,0,0],\
                                [0,0,matrix[2,2]/2+matrix[1,1]/2,0],\
                                [matrix[0,3]/2+matrix[3,0]/2,0,0,matrix[3,3]]]))
dat = []
for i in range(10000):
    initial_matrix = rand_PSDM(4)
    projected = projection(initial_matrix)
    parameters = [projected[0,0], projected[0,3], projected[1,1], projected[3,3]]
    projected = density_matrix(projected)
    projected.set(20000)
    dat.append((projected.data, parameters))
    if (i+1)%10 == 0:
        print(f'{i+1}/10000 done')

import pickle
with open('ML_proj_samps.dat', 'wb') as file:
    pickle.dump(dat, file)
        
# %% Data load & split
import pickle
with open('ML_proj_samps.dat', 'rb') as file:
    samples_load = pickle.load(file)     

samples_train = samples_new[:8000]
samples_val = samples_new[8000:]

# %% Dataset definition
class proj_dataset(Dataset):
    
    def __init__(self,samples):
        super(Dataset, self).__init__()
        self.samples = samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        hist, params = self.samples[idx]
        hist = torch.from_numpy(hist).float()
        params = torch.Tensor(params).float()
        return hist, params

class Net(nn.Module):
    
    def __init__(self, input=100, output=4):
        super().__init__()
        
        self.input_layer = nn.Linear(input, 50)
        self.output_layer = nn.Linear(20, output)
        self.hidden_layers = nn.Sequential(
            nn.Linear(50, 100),
            nn.Sigmoid(),
            nn.Dropout(p=0.5),
            nn.Linear(100,50),
            nn.Sigmoid(),
            nn.Dropout(p=0.3),
            nn.Linear(50,70),
            nn.Sigmoid(),
            nn.Dropout(p=0.3),
            nn.Linear(70,20),
            nn.Sigmoid(),
            nn.Dropout(p=0.3),
        )
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x   

#Trainig & validation loop
def fit(model, samples_train, samples_val, batch_size=10, lr=0.05, epochs=10, dataset = proj_dataset, criterion=nn.MSELoss()):    
    from torchmetrics import MeanSquaredError
    data_loader_train = DataLoader(dataset(samples_train), shuffle=True, batch_size=batch_size)
    data_loader_val = DataLoader(dataset(samples_val), shuffle=True, batch_size=batch_size)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr) 
    metrics = MeanSquaredError()

    for epoch in range(epochs):
        #Training
        model.train()
        for hist, params in data_loader_train:
            optimizer.zero_grad()
            pred = model(hist)
            loss = criterion(pred, params)
            loss.backward()
            optimizer.step()
            metrics(pred, params)
        train_loss = metrics.compute()
        metrics.reset()
        #Validation
        model.eval()
        with torch.no_grad():
            for hist, params in data_loader_val:
                pred = model(hist)
                metrics(pred, params)
        val_loss = metrics.compute()
        metrics.reset()
        
        print(f'Epoch {epoch+1}/{epochs}: training loss: {train_loss:.5f}, validation loss: {val_loss:.5f}')
    
    return model    
    
# %% Model training
model = fit(Net(), samples_train, samples_val, batch_size=50, lr=0.05, epochs=10)
# %% Comparison of model's prediction for two inputs
model.eval()
print(model(torch.from_numpy(samples_train[6][0]).float()))
print(model(torch.from_numpy(samples_train[16][0]).float()))

####### END OF CASE OF PROJECTED STATES #######               

####### CASE OF WERNER STATES #######
#%% Data load, reformat and split
df = pd.read_csv('werner_sample.csv', index_col='Unnamed: 0')
df_test = pd.read_csv('werner_test.csv', index_col='Unnamed: 0')
def df_to_wer_data(df):
    wer_data=[]
    for i in range(len(df)):
        wer_data.append((df.iloc[i].values[2:], df.iloc[i].values[:2]))
    return wer_data 
wer_data = df_to_wer_data(df)
wer_test = df_to_wer_data(df_test)
# %% Dataset and neural network definition

class werner_dataset(Dataset):
    
    def __init__(self, data, transform = torch.from_numpy):
        super(Dataset, self).__init__()
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        hist, params = self.data[idx]
        hist = self.transform(hist).float()
        params = self.transform(params).float()
        return hist, params
    
class WerNet(nn.Module):
    
    def __init__(self, input=100, output=4):
        super().__init__()
        
        self.input_layer = nn.Linear(input, 95)
        self.output_layer = nn.Linear(95, output)
        self.hidden_layers = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(95, 95),
            nn.Sigmoid(),
            nn.Linear(95,95),
            nn.Sigmoid(),

        )
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x   

# %% Model training & saving       
model_wer = fit(WerNet(input=100, output=2), wer_data[:8000], wer_data[8000:], dataset = werner_dataset, epochs=10)
torch.save(model_wer.state_dict(), 'model_wer.model')
# %% Model loading
model_loaded = WerNet(input=100, output=2)
model_loaded.load_state_dict(torch.load('model_wer.model'))
# %% Model testing
df_test = pd.read_csv('werner_test.csv', index_col='Unnamed: 0')
wer_test = df_to_wer_data(df_test)
evaluate_model(model_loaded, wer_test, dataset=werner_dataset)
# %% Finding the worst match
import torchmetrics
max_err = 0
worst_hist=None
worst_params=None
model_loaded.eval()
metrics = torchmetrics.MeanSquaredError() 
for hist, params in DataLoader(werner_dataset(wer_test), batch_size=1):
    pred = model_loaded(hist)
    err = metrics(pred, params)
    if err > max_err:
        max_err = err
        worst_hist = hist
        worst_pred = pred
        worst_params = params

alp, vis = worst_pred.detach().numpy()[0]
worst_dm = density_matrix(rho2(alp, vis))
plt.stairs(worst_hist.numpy()[0], fill=True)
plt.stairs(worst_dm.bins()['counts'])
plt.show()
# %% Finding the best match
import torchmetrics
min_err = 80
best_hist=None
best_params=None
model_loaded.eval()
metrics = torchmetrics.MeanSquaredError() 
for hist, params in DataLoader(werner_dataset(wer_test), batch_size=1):
    pred = model_loaded(hist)
    err = metrics(pred, params)
    if err < min_err:
        min_err = err
        best_hist = hist
        best_pred = pred
        best_params = params

alp, vis = best_pred.detach().numpy()[0]
best_dm = density_matrix(rho2(alp, vis))
plt.stairs(best_hist.numpy()[0], fill=True)
plt.stairs(best_dm.bins()['counts'])
plt.show()
