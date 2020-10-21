#%%
import sys
sys.path.append("/Users/huangguanlun/pyProjects/quotoFuture/")
sys.path.append("/Users/huangguanlun/pyProjects/quotoFuture/qfVenv/lib/python3.8/site-packages/")

#%%
import simContainer
import numpy as np
from scipy.linalg import cholesky, inv
# %%
S0Arr = [100, 120]
riskFreeRate = 0.05
timePeriod = 0.5
timePartitionCnt =1000
qArr = [0, 0]
sigmaArr = [0.03, 0.05]
corrMatrx = [[1, 0.4], [0.4, 1]]
simCnt = 200
repeatCnt = 100 

#%%
test = simContainer.pathSim(
    S0Arr= S0Arr,
    riskFreeRate= riskFreeRate,
    timePartitionCnt= timePartitionCnt,
    timePeriod= timePeriod,
    qArr= qArr,
    sigmaArr= sigmaArr,
    corrMatrx= corrMatrx,
    simCnt= simCnt,
    repeatCnt= repeatCnt
)

#%%
# check var_cov
test.var_cov()
np.cov(test.returnSeries[0][0], rowvar= False), test.covMtrx, test.returnSeries[0][0].mean(axis = 0)
np.corrcoef(test.returnSeries[0][0], rowvar=False)
#%%
test.returnSeries[0][0].mean(axis = 0)
# %%
# check mean
#%%
test.mu_delta_t_return

#%%
test.muShift()
test.returnSeries.mean(axis = (0, 1, 2))
# %%
# append lnS0 and cumsum
test.lnS0Append()
test.cumsum()

#%%
tail = test.timePartitionCnt - 20
test.stSeries[0, 0, :20, :], test.stSeries[0, 0, tail:, :]