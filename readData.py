

import numpy as np
import xlrd
import xlwt
import math
import random
from scipy.stats import weibull_min

# 导入参数
def readData(filepath):
    filename = filepath + 'data.xls'
    data = xlrd.open_workbook(filename)
    table = data.sheet_by_index(0)

    pI = np.int(table.row(1)[1].value)  #subsystem num
    sI = [i for i in range(pI)]         #I

    pJ = np.int(table.row(1)[3].value)  #max redundancy num
    sJ = [j for j in range(pJ)]         #J

    pT = np.int(table.row(1)[5].value)  #time horizon
    sT = [t for t in range(pT+1)]       #Tset
    sT_End = sT[:-1]                    #Tset\{T}
    sT_0 = sT[1:]                       #Tset\{0}

    pR = pT                             #max indicidual num
    sR = sT[:-1]                        #Rset
    sR_End = sR[:-1]                    #Rset\{R}
    sR_1 = sR[1:]                       #Rset\{1}

    pCP = [np.float32(table.row(i+1)[7].value) for i in sI]     #purchase cost
    pCM = [np.float32(table.row(i+1)[9].value) for i in sI]     #replacement cost
    pCF = [np.float32(table.row(i+1)[11].value) for i in sI]    #set-up cost

    pG = [np.int(table.row(i+1)[13].value) for i in sI]         #component state
    pD = np.int(table.row(1)[15].value)                         #sys demand
    pR0 = np.float32(table.row(1)[17].value)                    #R0

    LT_scale = [np.float32(table.row(i+1)[19].value) for i in sI]#lifetime distribution param
    LT_shape = [np.float32(table.row(i+1)[21].value) for i in sI]
    # mean of pLT
    pLT_mean = [math.ceil(weibull_min.mean(LT_shape[i], loc=0, scale=LT_scale[i])) for i in sI]

    pOmega = np.int(table.row(1)[23].value)                     #scenario num for evaluation
    sOmega = [w for w in range(pOmega)]                         #scenario set
    probOmega = [1/pOmega for w in sOmega]

    # ambiguity set of Tijr (support, mu, abs, greaterMu)
    pSuppLower = [np.float32(table.row(i+1)[25].value) for i in sI] #lower support
    pSuppUpper = [np.float32(table.row(i+1)[27].value) for i in sI] #upper support
    pMu = [np.float32(table.row(i+1)[29].value) for i in sI]        #meam
    pAbs = [np.float32(table.row(i+1)[31].value) for i in sI]       #E|Tijr-mu|
    pGreaterMu = [np.float32(table.row(i+1)[33].value) for i in sI] #Pr(Tijr>=mu)

    return pI, sI, pJ, sJ, pT, sT, sT_End, sT_0, pR, sR, sR_End, sR_1, pCP, pCM, pCF, pG, pD, pR0, LT_scale, LT_shape, pLT_mean, pOmega, sOmega, probOmega, pSuppLower, pSuppUpper, pMu, pAbs, pGreaterMu

# 生成 1个scenario - lifetime （pLT）
def pLTGeneration(sI, sJ, sR, LT_scale, LT_shape):
    """
     Generate All Scenarios for Component Lifetime
     :param sI                   :system set
     :param sJ                   :component set
     :param sR                   :repalcement set
     :param LT_scale, LT_shape   :lifetime distribution parameters
     :return:
     """
    pLT_key = [(i,j,r) for i in sI for j in sJ for r in sR]
    pLT_value = [0 for _ in pLT_key]
    pLT = dict(zip(pLT_key, pLT_value))
    for i in sI:
        for j in sJ:
            for r in sR:
                    pLT[i,j,r] = math.ceil(random.weibullvariate(LT_shape[i],LT_scale[i]))

    return pLT


# 生成 1个scenario - lifetime （pLT）
# 考虑不同的criterion：BC / WC / EM
def pLTGeneration_robust(criteria, sI, sJ, sR, pSuppLower, pSuppUpper, pMu, pAbs, pGreaterMu):
    if criteria == 'WC':
        return pLTGeneration_WC(sI, sJ, sR, pSuppLower, pSuppUpper, pMu, pAbs, pGreaterMu)
    elif criteria == 'BC':
        return pLTGeneration_BC(sI, sJ, sR, pSuppLower, pSuppUpper, pMu, pAbs, pGreaterMu)
    elif criteria == 'EM':
        return pLTGeneration_EM(sI, sJ, sR, pSuppLower, pSuppUpper, pMu, pAbs, pGreaterMu)
    elif criteria == 'random':
        return pLTGeneration_random(sI, sJ, sR, pSuppLower, pSuppUpper, pMu, pAbs, pGreaterMu)
    else:
        return False


# 生成 1个scenario - lifetime （pLT）
# 考虑为worst case (WC)
def pLTGeneration_WC(sI, sJ, sR, pSuppLower, pSuppUpper, pMu, pAbs, pGreaterMu):
    """
     Generate a sample for Component Lifetime
     """
    # 3点分布
    p1 = [1/2 * pAbs[i] / (pMu[i] - pSuppLower[i]) for i in sI]
    p2 = [1/2 * pAbs[i] / (pSuppUpper[i] - pMu[i]) for i in sI]

    pLT_key = [(i,j,r) for i in sI for j in sJ for r in sR]
    pLT_value = [0 for _ in pLT_key]
    pLT = dict(zip(pLT_key, pLT_value))
    for i in sI:
        for j in sJ:
            for r in sR:
                RM = random.random()
                if RM <= p1[i]:
                    pLT[i,j,r] = pSuppLower[i]
                elif RM <= p1[i]+p1[i]:
                    pLT[i,j,r] = pSuppUpper[i]
                else:
                    pLT[i,j,r] = pMu[i]

    return pLT


# 生成 1个scenario - lifetime （pLT）
# 考虑为best case (BC)
def pLTGeneration_BC(sI, sJ, sR, pSuppLower, pSuppUpper, pMu, pAbs, pGreaterMu):
    pLT_key = [(i, j, r) for i in sI for j in sJ for r in sR]
    pLT_value = [0 for _ in pLT_key]
    pLT = dict(zip(pLT_key, pLT_value))
    for i in sI:
        for j in sJ:
            for r in sR:
                RM = random.random()
                if RM <= pGreaterMu[i]:
                    pLT[i,j,r] = pMu[i] + 1/2*pAbs[i]/pGreaterMu[i]
                else:
                    pLT[i,j,r] = pMu[i] - 1/2*pAbs[i]/(1-pGreaterMu[i])

    return pLT


# 生成 1个scenario - lifetime （pLT）
# 考虑为expected mean (EM)
def pLTGeneration_EM(sI, sJ, sR, pSuppLower, pSuppUpper, pMu, pAbs, pGreaterMu):
    """也是说，Tijr是固定的值pMu[i]"""
    pLT_key = [(i, j, r) for i in sI for j in sJ for r in sR]
    pLT_value = [0 for _ in pLT_key]
    pLT = dict(zip(pLT_key, pLT_value))
    for i in sI:
        for j in sJ:
            for r in sR:
                pLT[i,j,r] = pMu[i]

    return pLT


# 生成 1个scenario - lifetime （pLT）
# 考虑为random
def pLTGeneration_random(sI, sJ, sR, pSuppLower, pSuppUpper, pMu, pAbs, pGreaterMu):
    """Tijr在support中随机生成"""
    pLT_key = [(i, j, r) for i in sI for j in sJ for r in sR]
    pLT_value = [0 for _ in pLT_key]
    pLT = dict(zip(pLT_key, pLT_value))
    for i in sI:
        for j in sJ:
            for r in sR:
                pLT[i,j,r] = math.ceil(random.uniform(pSuppLower[i], pSuppUpper[i]))

    return pLT


if __name__ == '__main__':
    filepath = 'data//'

    # read data
    pI, sI, pJ, sJ, pT, sT, sT_End, sT_0, pR, sR, sR_End, sR_1, pCP, pCM, pCF, pG, pD, pR0, LT_scale, LT_shape, \
                    pLT_mean, pOmega, sOmega, probOmega, pSuppLower, pSuppUpper, pMu, pAbs, pGreaterMu = readData(filepath)

    # generate lifetime for one scenario
    pLT = pLTGeneration(sI, sJ, sR, LT_scale, LT_shape)

    # generate lifetime for one scenario under different criterion (WC)
    criteria = 'BC'  # 'WC' or 'BC' or 'EM' -- robust criteria
    pLT = pLTGeneration_robust(criteria, sI, sJ, sR, pSuppLower, pSuppUpper, pMu, pAbs, pGreaterMu)








