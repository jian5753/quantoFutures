import numpy as np
from scipy.linalg import cholesky, inv

class z0Container():
    def __init__(self, shape, varCntAxis):
        self.shape = shape
        self.varCntAxis = varCntAxis
        self.container = np.random.normal(
            loc = 0.0,
            scale = 1.0,
            size = self.shape
        )
    
    def antithetic(self, get=False):
        for i, trial in enumerate(self.container):
            for j, simPath in enumerate(self.container[i]):
                halfPoint = int(self.shape[2]/2) + 1
                firstHalf = simPath[:halfPoint]
                secondHalf = firstHalf * (-1)
                temp = np.concatenate([firstHalf, secondHalf], axis = 0)
                self.container[i][j] = temp[:self.shape[2]]
        if get:
            return self.container

    def invCholesky(self, get=False):
        self.antithetic()
        for i, trial in enumerate(self.container):
            for j, simPath in enumerate(self.container[i]):
                cov = np.cov(simPath, rowvar=False)
                covUtriInv = inv(cholesky(cov, lower=False))
                self.container[i][j] = simPath.dot(covUtriInv)
        if get:
            return self.container

    def reset(self):
        self.container = np.random.normal(
            loc = 0.0,
            scale = 1.0,
            size = self.shape
        )
        self.invCholesky(get= False)
    

class pathSim():
    def __init__(self, S0Arr, riskFreeRate, timePeriod, timePartitionCnt, qArr, sigmaArr, corrMatrx, simCnt, repeatCnt):
        self.lnS0Arr = np.log(np.array(S0Arr))
        self.assetCnt = self.lnS0Arr.shape[0]
        self.riskFreeRate = riskFreeRate
        self.timePeriod = timePeriod
        self.timePartitionCnt = timePartitionCnt
        self.qArr = np.array(qArr)
        self.sigmaArr = np.array(sigmaArr)
        self.corrMtrx = np.array(corrMatrx)
        self.simCnt = simCnt
        self.repeatCnt = repeatCnt
        self.covMtrx = self.corrMtrx * np.tensordot(self.sigmaArr.reshape(1, -1), self.sigmaArr.reshape(1, -1), axes=(0,0)) 

        self.Z0Container = z0Container([self.repeatCnt, self.simCnt, self.timePartitionCnt - 1, self.assetCnt], varCntAxis= 3)
        self.Z0 = self.Z0Container.invCholesky(get= True)

        self.returnSeries = np.zeros_like(self.Z0)

        self.mu_delta_t_return = (self.riskFreeRate - self.qArr - np.power(self.sigmaArr, 2) / 2) * (self.timePeriod / self.timePartitionCnt)


    def var_cov(self):
        utri = cholesky(self.covMtrx)
        # var-cov
        self.returnSeries = np.tensordot(self.Z0, utri, axes = ([3], [0]))

    def muShift(self):
        # mu shift
        self.returnSeries += self.mu_delta_t_return

    def lnS0Append(self):
        # append lnS0
        self.lnS0ArrPanel = self.lnS0Arr * np.ones((self.repeatCnt, self.simCnt, 1, self.assetCnt))
        self.lnStSeries = np.concatenate((self.lnS0ArrPanel, self.returnSeries), axis= 2)
    
    def cumsum(self):
        # cumsum
        self.lnStSeries = self.lnStSeries.cumsum(axis= 2)
        self.stSeries = np.exp(self.lnStSeries)



