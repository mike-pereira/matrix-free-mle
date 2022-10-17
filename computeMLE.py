#%%

import time
import numpy as np
import gstlearn as gl
import nlopt as nl

#%%

## Class for MLE computation

class computeMLE :
    
    def __init__(self,grid,dat,mesh,structParam,nHuntch=1,tolBypass=1e-3,verbose=True):
        self.grid = grid
        self.dat = dat
        self.mesh = mesh
        self.structParam=structParam
        self.nHuntch = nHuntch
        self.seedHuntch=int(np.random.uniform(1,10**4))
        self.Nparam=0
        self.iterCount=0
        self.verbose=verbose
        self.tolBypass=tolBypass
        self.printLog=False
        self.logFileName=""
        self.timeBegin=0
        self.OptAlg=0
        

    ## Parametrization of tau^2 parameter
    def paramTau2(self,pTau2):
        return np.exp(pTau2)

    ##  Create covariance model
    def createModel(self,x,canonicalParam=False):
        self.model = gl.Model.createFromParam(gl.ECov.MARKOV,ranges=self.structParam[0],angles = self.structParam[1],
                                     sill=self.structParam[2],flagRange=False)
        cova = self.model.getCova(0)
        
        if canonicalParam:
            cova.setMarkovCoeffs([x[0],x[1],x[2],x[3]])
        else :
            cova.setMarkovCoeffsBySquaredPolynoms([x[0],x[1]],[x[2],x[3]],0.001)
        
        self.model.addCovFromParam(gl.ECov.NUGGET,sill=self.paramTau2(x[4]))
    
    ## Create SPDE object        
    def createSPDE(self):
        self.spde = gl.SPDE()
        self.spde.init(self.model,self.grid,self.dat,
                       gl.ESPDECalcMode.LIKELIHOOD,self.mesh)
        self.spde.compute()
        
    ## Compute negative log-likelihood
    def computeLogLk(self,x,canonicalParam=False) :
        
        ## Bypass computation if change in parameter value too small
        err=(np.array(self.paramCurrent-x)**2).max()**0.5
        if err<self.tolBypass:
            self.cb(self.paramCurrent,self.lossCurrent)
            return self.lossCurrent
            
        self.createModel(x,canonicalParam)
        self.createSPDE()
        res = -self.spde.computeLogLike(self.nHuntch,self.seedHuntch)

        return res

    ## Initialize log file
    def initLogFile(self):
        with open(self.logFileName, 'w') as file:
            file.write(','.join(['Iteration','LogLk',','.join(['Coef Markov ' +str(i) for i in np.arange(0,self.Nparam-1,dtype=int)]),'Coef tau','Time (min)'])+'\n')        
    
    ## Define Calllbacks
    def cb(self,paramcur,valcur):
        self.iterCount+=1
        timeElapsed=(time.time()-self.timeBegin)/60

        if self.verbose:
            print("-----------------------------------------------------")        
            print("iteration ", str(self.iterCount),"(Time Elasped = ",timeElapsed," min)" )
            print("Loglikelihood: " + str(valcur))            
            print("Current parameter values:",paramcur)
            
        if(self.printLog):
            with open(self.logFileName, 'a') as file:
                file.write(','.join([str(self.iterCount),str(valcur),','.join(map(str, paramcur[:4])),str(paramcur[4]),str(timeElapsed)])+'\n')       
                

    ## Define cost function for minimizer
    def costFunc(self,x,grad):  
        ## Compute negative log-likelihood          
        res=self.computeLogLk(x) 
        ## Save current parameter value and cost value  
        self.paramCurrent=np.copy(x)
        self.lossCurrent=res
        ## Callback
        self.cb(self.paramCurrent,self.lossCurrent)
            
        return res
    
    ## Minimization
    def minimize(self,paramInit,nbEvals,bounds=None,printLog=False,logFileName="log.csv"):
        
        ## Initialize parameter
        self.iterCount=0
        self.Nparam=len(paramInit)
        self.paramCurrent=np.zeros(len(paramInit))+np.inf
        self.lossCurrent=np.inf
        
        ## Log file initialization
        self.printLog=printLog
        self.logFileName=logFileName
        if(self.printLog):
            self.initLogFile()
        
        ## Initialize optimizer
        if self.OptAlg==0:
            print("Optimzer: COBYLA")
            optimizer = nl.opt(nl.LN_COBYLA, len(paramInit))
        elif self.OptAlg==1:
            print("Optimzer: BOBYQA")
            optimizer = nl.opt(nl.LN_BOBYQA, len(paramInit))
        else:
            print("Optimzer: SBPLX")
            optimizer = nl.opt(nl.LN_SBPLX, len(paramInit))
        
        ## Set Objective
        optimizer.set_min_objective(self.costFunc)
        if bounds is not None:
            optimizer.set_lower_bounds(bounds[0])
            optimizer.set_upper_bounds(bounds[1])
        optimizer.set_maxeval(nbEvals)
        
        # Perform optimization
        self.timeBegin=time.time()
        res = optimizer.optimize(paramInit)
        
        return [res, optimizer.last_optimum_value()]
    