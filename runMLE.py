#########################
# Run Max Likelihood
#########################


# %% Imports
import matplotlib.pyplot as plt
import gstlearn as gl
import numpy as np
from computeMLE import computeMLE

# %% Initialize grids

## Grid param for simulating data and displays: Grid [-5, 45] x [-5, 45]
x0 = [-5.,-5.]
dx = [.125,.125]
nx = [400,400]

## Grid params for internal computation in likelihood maximization
nxCoarse = [125,125]
dxCoarse = [50/nxCoarse[0],50/nxCoarse[1]]

## Create grids
dbCoarse = gl.DbGrid.create(nxCoarse, dxCoarse, x0)
dbTrue = gl.DbGrid.create(nx, dx, x0)

## Define data in [0,40] x [0, 40] (to mitigate edge effects)
np.random.seed(123445)
ndat = 15000
tau2True = 0.01
dbDat = gl.Db()
dbDat["x1"]=40. * np.random.uniform(size=ndat)
dbDat["x2"]=40. * np.random.uniform(size=ndat)
dbDat.setLocators(["x1","x2"],gl.ELoc.X)


# %% Covariance model of data

## Sill parameter
sill = 1.

## Anisotropy parameters (none)
ranges = [1. , 1.]
angles = [0,0]

## True coefs of Markov model in canonical form (i.e. P(x)=\sum_{i} coeffs[i] * x**i)
coeffsTrue = [1,-0.75,-0.75,1] 

## Create covaraince model
modelTrue = gl.Model.createFromParam(gl.ECov.MARKOV,ranges=ranges,angles = angles,
                                     sill=sill,flagRange=False)
cova = modelTrue.getCova(0)
cova.setMarkovCoeffs(coeffsTrue)


# %% Mesh definitions

mesh = gl.MeshETurbo(dbTrue)
meshCoarse = gl.MeshETurbo(dbCoarse)


# %% Create simulated data

## Create SPDE object
spdeTrue = gl.SPDE()
spdeTrue.init(modelTrue,dbTrue,None,gl.ESPDECalcMode.SIMUNONCOND,mesh) 

## Compute simulation
seedSim=1140
spdeTrue.compute(1,seedSim)

## Migrate simulation to db objects
spdeTrue.query(dbTrue)
spdeTrue.query(dbDat)


# %% Create selection to remove edges

selmin=np.logical_and(dbTrue["x1"]>=0,dbTrue["x2"]>=0)
selmax=np.logical_and(dbTrue["x1"]<=40,dbTrue["x2"]<=40)
sel = np.logical_and(selmin,selmax)
dbTrue.addSelection(sel)


# %% Create observed data

## Create observed data by adding measurement noise
np.random.seed(seedSim+1)
dbDat["dat"] = dbDat["spde*"][:,0] + np.sqrt(tau2True) * np.random.normal(size=ndat)


# %% Initialze MLE object

mle = computeMLE(dbCoarse,dbDat,meshCoarse,
                 structParam=[ranges,angles,sill],
                 tolBypass=1e-4,verbose=True)


#%% Initial guess
  
paramInit = [0,0,0,0,0]

  
#%% Compute MLE

## Bounds for parameters
lb=[-10,-10,-10,-10,-10]
ub=[10,10,10,10,2]

## Compute 

### First pass using one random vector for Huntchinson computation
mle.nHuntch=1
res=mle.minimize(paramInit,nbEvals=200,bounds=[lb,ub],
                    printLog=True,logFileName='/home/mpereira/Documents/Work/dev/Py/Papier JDS/Final/Log_'+str(seedSim)+'.csv')

### Second pass using 10 random vectors for Huntchinson computation
mle.nHuntch=10
res=mle.minimize(res[0],nbEvals=50,bounds=[lb,ub],
                    printLog=True,logFileName='/home/mpereira/Documents/Work/dev/Py/Papier JDS/Final/LogFin_'+str(seedSim)+'.csv') 

## Solution
paramcur=res[0]
print("Loss :"+str(res[1])+"/ Parameters: "+','.join(map(str, paramcur[:4]))+","+str(mle.paramTau2(paramcur[4])))


