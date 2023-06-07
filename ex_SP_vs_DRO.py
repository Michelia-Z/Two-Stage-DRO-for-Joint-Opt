# out-of-sample experiment
import random

import readData, maintenanceModel, DeterministicModel, printSol
import RAMPmodel, initialSolution, Mobj
import RAMPmodel_robust, initialSolution_robust, Mobj_robust

import math
import numpy as np
import xlwt

# read data
filepath = 'data//'
pI, sI, pJ, sJ, pT, sT, sT_End, sT_0, pR, sR, sR_End, sR_1, pCP, pCM, pCF, pG, pD, pR0, LT_scale, LT_shape, pLT_mean, \
        pOmega_eval, sOmega_eval, probOmega, pSuppLower, pSuppUpper, pMu, pAbs, pGreaterMu = readData.readData(filepath)
pOmega_ALNS = 100
penalty = 1000


# 考虑不同criteria(WC/BC/EM)下的robust RAMP的求解
def solveRAMP_robust(criteria):
    # EM-RAMP是确定性问题，直接CPLEX求解
    if criteria == 'EM':
        pLT = readData.pLTGeneration_robust(criteria, sI, sJ, sR, pSuppLower, pSuppUpper, pMu, pAbs, pGreaterMu)
        mdl = DeterministicModel.RAMPmodel(sI, sJ, pT, sT, sR, pCP, pCM, pCF, pG, pD, pLT, pR0)
        mdl.solve()
        xBest = [int(sum(mdl.xx_vars[i,j].solution_value for j in sJ)) for i in sI]
        objBest = mdl.objective_value
    # WC/BC-RAMP是随机优化问题，用ALNS求解
    else:
        # intial sol
        xInitial = initialSolution_robust.feasibleSol(criteria, sI, pJ, sJ, pT, sT, sT_End, sT_0, sR, sR_End, sR_1, pG, pD,
                                                      pR0, sOmega_eval, pSuppLower, pSuppUpper, pMu, pAbs, pGreaterMu)
        # ALNS solve robust RAMP
        print('\n----------- robust RAMP', criteria, '-------------')
        objInitial = Mobj_robust.objRAMP(criteria, xInitial, sI, sJ, pT, sT, sT_End, sT_0, sR, sR_End, sR_1, pCP, pCM, pCF,
                                         pG, pD, pR0, pOmega_ALNS, pSuppLower, pSuppUpper, pMu, pAbs, pGreaterMu)
        T0_SA = 0.03 * objInitial / math.log(2)  # initial temperature in SA
        theta_SA = (math.log(2) / math.log(1e5)) ** (1 / 1000)
        alns = RAMPmodel_robust.ALNS(T0_SA, theta_SA, xInitial, objInitial, pOmega_ALNS, criteria)
        alns.run()
        xBest = alns.xBest
        objBest = alns.objBest

    return xBest, objBest


# out-of-sample exp for robust RAMP
def outOfSample_robust(criteriaSet, xBestSet, Num_outOfSample):
    """
    :param criteria: WC/BC -- robust criteria
    :param xx_values: decision value of xx
    :param Num_outOfSample: sample num for our-of-sample
    :return:
    """

    OBJ_OOS_set = [[] for _ in criteriaSet]
    for w in range(Num_outOfSample):
        print('out-of-sample=', w+1, '/', Num_outOfSample)

        # generate a sample for out-of-sample exp
        rm = random.random()
        if rm <= 3/4:
            pLT = readData.pLTGeneration_robust('WC', sI, sJ, sR, pSuppLower, pSuppUpper, pMu, pAbs, pGreaterMu)
        else:
            pLT = readData.pLTGeneration_robust('random', sI, sJ, sR, pSuppLower, pSuppUpper, pMu, pAbs, pGreaterMu)

        # evaluate all solutions (xBestSet)
        for k in range(len(xBestSet)):
            xBest = xBestSet[k]
            xx_values = xTrans(xBest, sI, sJ)   # 将xi转换成xij的形式 [2,1,1] -> [[1,1,0], [1,0,0], [1,0,0]]
            # evaluate sol xx_values
            mdl = maintenanceModel.RAMPmodel(xx_values, sI, sJ, pT, sT, sR, pCP, pCM, pCF, pG, pD, pLT, pR0)
            ms = mdl.solve()
            if not ms:
                obj = 1e+5 # 如果无解 -> big enough
            else:
                obj = mdl.objective_value
            OBJ_OOS_set[k].append(obj)

    return OBJ_OOS_set


# 转换变量x的表示形式
def xTrans(x, sI, sJ):
    """
    x=[2,2,2] --> y=[[1,1,0],[1,1,0],[1,1,0]]
    """
    idx = [(i,j) for i in sI for j in sJ]
    values = [0 for _ in idx]
    y = [[0 for j in sJ] for i in sI]
    for i in sI:
        for j in range(x[0]):
            y[i][j] = 1
    return y


# out-of-sample 统计信息
def statisticInfo(criteriaSet, xBestSet, objBestSet, OBJ_OOS_set):
    """
    :param criteria:        WC/BC/EM
    :param xBestSet:        set of optimal sol
    :param objBestSet:      set ofoptimal obj
    :param OBJ_OOS_set:     set of out-of-sample obj for total cost
    :return: info - dict
    info['criteria']        -- robust criteria
    info['x']               -- optimal sol for RAMP
    info['obj']             -- optimal obj for RAMP
    info['obj-firstStage']  -- first-stage obj of x
    ===================================================
    ================= out-of-sample ===================
    ===================================================
    info['OBJ_OOS']         -- out-of-sample, total obj
    info['min']             -- min(second-stage obj)
    info['max']             -- max(second-stage obj)
    info['aver']            -- mean(second-stage obj)
    info['dec']
    info['qua']
    info['std']             -- std(second-stage obj)
    info['infeasible']      -- infeasible(second-stage problem)

    return infoSet = [info1, info2, ...]
    """

    InfoSet = []
    for k in range(len(criteriaSet)):   # 遍历所有的criteria对应的sol
        OBJ_OOS = OBJ_OOS_set[k]
        xBest = xBestSet[k]
        objBest = objBestSet[k]
        criteria = criteriaSet[k]

        # 统计信息
        OBJ_OOS = np.array(OBJ_OOS)
        obj_1 = sum(pCP[i] * xBest[i] for i in sI)  # first-stage obj

        # --- OBJ_OOS total cost --> second-stage cost
        infeasible = 0
        for m in range(len(OBJ_OOS)):
            if OBJ_OOS[m] == 1e+5:
                OBJ_OOS[m] = penalty    # infeasible解替换成penalty
                infeasible += 1
            else:
                OBJ_OOS[m] = OBJ_OOS[m] - obj_1

        # statistical info for second-stage obj
        min = OBJ_OOS.min()
        max = OBJ_OOS.max()
        aver = OBJ_OOS.mean()
        std = OBJ_OOS.std()
        dec = np.percentile(OBJ_OOS, 75)
        qua = np.percentile(OBJ_OOS, 90)
        info = {'criteria':criteria, 'x':xBest, 'obj':objBest, 'obj-firstStage':obj_1, 'min':min, 'max':max, 'aver':aver, 'dec':dec, 'qua':qua,'std':std, 'OBJ_OOS':OBJ_OOS, 'infeasible':infeasible}
        InfoSet.append(info)
    return InfoSet



if __name__ == '__main__':
    Num_outOfSample = 100
    criteriaSet = ['WC', 'BC', 'EM']

    for criteria in criteriaSet:
        # solve
        xBest, objBest = solveRAMP_robust(criteria)
        # write sol
        printSol.printXObj(criteria, xBest, objBest)

    # out-of-sample experiment
    xBestSet = []
    objBestSet = []
    for criteria in criteriaSet:
        xBest, objBest = printSol.readXObj(criteria)
        xBestSet.append(xBest)
        objBestSet.append(objBest)

    OBJ_OOS_set = outOfSample_robust(criteriaSet, xBestSet, Num_outOfSample)
    InfoSet = statisticInfo(criteriaSet, xBestSet, objBestSet, OBJ_OOS_set)
    printSol.printInfo(InfoSet)