import numpy as np

class simContainer():
    def __init__(self, s0Arr, muArr, sigmaArr, corrMatrx, timePeriod, simPathCnt, trialCnt):
        self.s0Arr = np.array(s0Arr) 
        self.muArr = np.array(muArr)
        self.sigmaArr = np.array(sigmaArr)
        self.corrMatrx = np.array(corrMatrx)
        self.timePeriod = timePeriod
        self.simPathCnt = simPathCnt
        self.trialCnt = trialCnt

        self.assetCnt = self.muArr.shape[0]
        self.simResult = np.zeros((self.assetCnt, timePeriod, simPathCnt * trialCnt))

    def sim(self):
        raise KeyboardInterrupt
