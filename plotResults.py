#%%
import numpy as np
import matplotlib.pyplot as plt
import gstlearn as gl


#%%  Functions

## Function to compute covariance function from model
def computeCovFunc(x,hmax,canonical=False): 
    
    N = 2**8

    mH1 = hmax
    mH2 = mH1
    
    model = gl.Model.createFromParam(gl.ECov.MARKOV,ranges=[1. , 1.],angles = [0. , 0.],
                                     sill=1,flagRange=False)    
    cova = model.getCova(0)
    if canonical:
        cova.setMarkovCoeffs(x)
    else:
        cova.setMarkovCoeffsBySquaredPolynoms([x[0],x[1]],
                                      [x[2],x[3]],0.001)
    
    result = np.array(cova.evalCovFFT([mH1,mH2],N).getValues())
    resTrue = np.array(result).reshape((N,N))

    X1 = np.linspace(-mH1,mH1,N)
    hvec=X1[int(len(X1)/2):]
    cov=resTrue[int(N/2),:][int(len(X1)/2):]
    
    return [hvec,cov]


## Function to compute coefficients in canonical parametrization
def computeMarkovCoeffs(x): 

    model = gl.Model.createFromParam(gl.ECov.MARKOV,ranges=[1. , 1.],angles = [0. , 0.],
                                     sill=1,flagRange=False)    
    cova = model.getCova(0)
    cova.setMarkovCoeffsBySquaredPolynoms([x[0],x[1]],
                                      [x[2],x[3]],0.001)
    
    return cova.getMarkovCoeffs()


#%%

## Maximum lag
hmax=25

## True covariance
paramTrue=[1,-0.75,-0.75,1,0.01]
covValTrue=computeCovFunc(paramTrue,hmax,canonical=True)

## Estimated covariance
paramBest=[-0.73930176,  0.10571413,  1.60599707, -1.61223783, -4.66236796]
covVal=computeCovFunc(paramBest,hmax,canonical=False)

## Plot
plt.plot(covValTrue[0],covValTrue[1],label="True",color="red",linestyle="--")
plt.plot(covVal[0],covVal[1],label="Estimated")
plt.legend()
plt.show()

# %%
