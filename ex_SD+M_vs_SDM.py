"""
EXP:    RAP+M v.s. RAMP
"""

import random

import readData, maintenanceModel, DeterministicModel, printSol
import RAMPmodel, initialSolution, Mobj, RAPmodel

import math
import numpy as np
import xlwt

# read data
filepath = 'data//'
pI, sI, pJ, sJ, pT, sT, sT_End, sT_0, pR, sR, sR_End, sR_1, pCP, pCM, pCF, pG, pD, pR0, LT_scale, LT_shape, pLT_mean, \
        pOmega_eval, sOmega_eval, probOmega, pSuppLower, pSuppUpper, pMu, pAbs, pGreaterMu = readData.readData(filepath)
pOmega_ALNS = 100


# solve RAP+M
def RAP_M(RAPsampleNum, pTlimit):
    # solve RAP
    xFirstStage, objFirstStage = RAPmodel.solveRAP(RAPsampleNum, pTlimit)
    # solve M
    xx_trans = [int(sum(x)) for x in xFirstStage]
    objTotal = Mobj.objRAMP(xx_trans, sI, sJ, pT, sT, sT_End, sT_0, sR, sR_End, sR_1, pCP, pCM, pCF, pG, pD,
                                  pR0, LT_scale, LT_shape, pOmega_ALNS)
    return xx_trans, objTotal


# solve RAMP
def RAMP():
    # intial sol
    xInitial = initialSolution.feasibleSol(sI, pJ, sJ, pT, sT, sT_End, sT_0, sR, sR_End, sR_1, pG, pD, pR0, LT_scale,
                                           LT_shape, sOmega_eval)
    # ALNS solve robust RAMP
    objInitial = Mobj.objRAMP(xInitial, sI, sJ, pT, sT, sT_End, sT_0, sR, sR_End, sR_1, pCP, pCM, pCF, pG, pD, pR0,
                              LT_scale, LT_shape, pOmega_ALNS)
    T0_SA = 0.03 * objInitial / math.log(2)  # initial temperature in SA
    theta_SA = (math.log(2) / math.log(1e5)) ** (1 / 1000)
    alns = RAMPmodel.ALNS(T0_SA, theta_SA, xInitial, objInitial, pOmega_ALNS)
    alns.run()
    xBest = alns.xBest
    objBest = alns.objBest
    return xBest, objBest


def ex_joint(RAPsampleNum, pTlimit):
    # solve RAP+M
    xBest1, objBest1 = RAP_M(RAPsampleNum, pTlimit)

    # solve RAMP
    print('\n----------- RAMP', '-------------')
    xBest2, objBest2 = RAMP()

    xBestSet = [xBest1, xBest2]
    objBestSet = [objBest1, objBest2]
    printSol.printXObjSet('RAP+M vs RAMP R0=' + str(pR0) + ' t=' + str(pTlimit) + '.xls', xBestSet, objBestSet)


if __name__ == '__main__':
    RAPsampleNum = 100
    pTlimit = 2
    ex_joint(RAPsampleNum, pTlimit)