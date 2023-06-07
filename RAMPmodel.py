"""
RAMP ALNS求解
"""

import math, random, copy
import matplotlib.pyplot as plt
import readData, Mobj, initialSolution
from scipy.stats import weibull_min

# read data
filepath = 'data//'
pI, sI, pJ, sJ, pT, sT, sT_End, sT_0, pR, sR, sR_End, sR_1, pCP, pCM, pCF, pG, pD, pR0, LT_scale, LT_shape, pLT_mean, \
        pOmega_eval, sOmega_eval, probOmega, pSuppLower, pSuppUpper, pMu, pAbs, pGreaterMu = readData.readData(filepath)

# parameter for SA
rho_ALNS = 0.5          #reaction factor in searching neighborhood
operNum_ALNS = 8             #num of operator

segmentLen_ALNS = 2     #segment的长度
segmentNum_ALNS = 20     #segment的个数（停止准则）
pi1_ALNS = 15
pi2_ALNS = 10
pi3_ALNS = 1


class ALNS:
    '''
    :param T0           :initial temperature
    :param xInitial     :initial feasible solution
    :param pOmega_ALNS  :num of sample to calculate obj
    :param weightOper   :weight of operation in searching neighborhood
    '''
    def __init__(self, T0_SA, theta_SA, xInitial, objInitial, pOmega_ALNS):
        self.T = T0_SA
        self.theta_SA = theta_SA
        self.x = xInitial                       # 解x
        self.obj = objInitial                   # obj
        self.pOmega_ALNS = pOmega_ALNS          # ALNS中评估obj的样本大小
        self.weightOper = [1 for w in range(operNum_ALNS)]      # 算子的权重
        self.objBest = objInitial                  # 截止到目前，得到的最优值
        self.xBest = self.x                     # 截止到目前，得到的最优解
        self.history = {'x': [], 'obj': []}     # 记录每一次的解和值


    # generate new solutions in the neighborhood
    # return xNew & the used operator
    def generationNewSol(self):
        # ------------ select Operator -----------------------
        # 按照self.weightOper的权重来随机挑选算子，进行邻域搜索
        weight = self.weightOper
        probOper = [w / sum(weight) for w in weight]  # 算子的选择概率
        labelOper = [sum(probOper[:i + 1]) for i in range(operNum_ALNS)]  # 累计概率
        RM = random.random()
        operUsed = -1
        for i in range(operNum_ALNS):
            if RM <= labelOper[i] - 1e-5:
                operUsed = i        # 被选中的算子
                break
        # ------------ conduct Operator -----------------------
        xNew = []       # 生成的新解
        if operUsed == 0:
            xNew = self.operator_remove_random()
        elif operUsed == 1:
            xNew = self.operator_remove_maxNum()
        elif operUsed == 2:
            xNew = self.operator_remove_minR()
        elif operUsed == 3:
            xNew = self.operator_remove_maxC()
        elif operUsed == 4:
            xNew = self.operator_insert_random()
        elif operUsed == 5:
            xNew = self.operator_insert_minNum()
        elif operUsed == 6:
            xNew = self.operator_insert_maxR()
        elif operUsed == 7:
            xNew = self.operator_insert_minC()

        return xNew, operUsed

    ############################################################################
    #                               operators
    ############################################################################
    def operator_remove_random(self):
        # 随机选择一个子系统，减少一个组件
        subsystemSeleted = random.randint(1, pI) -1
        xNew = copy.deepcopy(self.x)
        xNew[subsystemSeleted] = xNew[subsystemSeleted] - 1
        return xNew

    def operator_insert_random(self):
        # 随机选择一个子系统，增加一个组件
        subsystemSeleted = random.randint(1, pI) -1
        xNew = copy.deepcopy(self.x)
        xNew[subsystemSeleted] = xNew[subsystemSeleted] + 1
        return xNew

    def operator_remove_maxNum(self):
        # 选择冗余最大的子系统，减少一个组件
        redundancy = self.x
        maxNum = max(redundancy)
        subsystemSeleted = redundancy.index(maxNum)
        xNew = copy.deepcopy(self.x)
        xNew[subsystemSeleted] = xNew[subsystemSeleted] - 1
        return xNew

    def operator_insert_minNum(self):
        # 选择冗余最小的子系统，增加一个组件
        redundancy = self.x
        minNum = min(redundancy)
        subsystemSeleted = redundancy.index(minNum)
        xNew = copy.deepcopy(self.x)
        xNew[subsystemSeleted] = xNew[subsystemSeleted] + 1
        return xNew

    def operator_remove_minR(self):
        # 选择可靠性最小的组件，减少一个
        reliability = [1-weibull_min.cdf(pT, LT_shape[i], loc=0, scale=LT_scale[i]) for i in sI]
        minNum = min(reliability)
        subsystemSeleted = reliability.index(minNum)
        xNew = copy.deepcopy(self.x)
        xNew[subsystemSeleted] = xNew[subsystemSeleted] - 1
        return xNew

    def operator_insert_maxR(self):
        # 选择可靠性最大的组件，增加一个
        reliability = [1-weibull_min.cdf(pT, LT_shape[i], loc=0, scale=LT_scale[i]) for i in sI]
        maxNum = max(reliability)
        subsystemSeleted = reliability.index(maxNum)
        xNew = copy.deepcopy(self.x)
        xNew[subsystemSeleted] = xNew[subsystemSeleted] + 1
        return xNew

    def operator_remove_maxC(self):
        # 选择维修成本最大的组件，减少一个
        maxNum = max(pCM)
        subsystemSeleted = pCM.index(maxNum)
        xNew = copy.deepcopy(self.x)
        xNew[subsystemSeleted] = xNew[subsystemSeleted] - 1
        return xNew

    def operator_insert_minC(self):
        # 选择维修成本最小的组件，增加一个
        minNum = min(pCM)
        subsystemSeleted = pCM.index(minNum)
        xNew = copy.deepcopy(self.x)
        xNew[subsystemSeleted] = xNew[subsystemSeleted] + 1
        return xNew
        ############################################################################

    def acceptance(self, f, f_new):
        if f_new <= f:
            return 1
        else:
            p = math.exp((f-f_new)/self.T)
            if random.random() < p:
                return 1
            else:
                return 0

    def run(self):
        #记录已经找到过的解sol
        xObtained = []
        xObtained.append(self.x)
        xObtained_obj = []
        xObtained_obj.append(self.obj)

        #记录每个segment算子的使用情况(分数score、使用次数num)
        operInfo_key = [(k, 'score') for k in range(operNum_ALNS)] + [(k, 'num') for k in range(operNum_ALNS)]
        operInfo_val = [0 for a in operInfo_key]
        operInfo = dict(zip(operInfo_key, operInfo_val))
        recordOperInfo = [copy.deepcopy(operInfo)]   #记录整个算法的算子使用情况
        recordWeight = [copy.deepcopy(self.weightOper)]

        step = 0
        while step < segmentNum_ALNS:
            step += 1
            count = 0
            # segment迭代
            while count < segmentLen_ALNS:
                count += 1
                xNew, operUsed = self.generationNewSol()
                operInfo[operUsed, 'num'] += 1  # 算子使用次数+1

                # 如果xNew曾经得到过，那么就不再计算
                if xNew in xObtained:
                    objNew = xObtained_obj[xObtained.index(xNew)]
                else:
                    objNew = Mobj.objRAMP(xNew, sI, sJ, pT, sT, sT_End, sT_0, sR, sR_End, sR_1, pCP, pCM, pCF, pG, pD, pR0, LT_scale, LT_shape, self.pOmega_ALNS)
                    xObtained.append(xNew)
                    xObtained_obj.append(objNew)

                print('segment=', step, 'count=', count, 'obj=', self.obj, 'best=', self.objBest)
                print('\t\t\tx=', self.x, 'xNew=', xNew)

                # 判断接受准则
                if self.acceptance(self.obj, objNew):
                    self.x = xNew
                    objOld = self.obj
                    self.obj = objNew
                    # 更新weight
                    if self.obj < self.objBest - 1e-5:
                        self.objBest = self.obj
                        self.xBest = self.x
                        operInfo[operUsed, 'score'] += pi1_ALNS     #算子分数+pi1
                    elif self.obj < objOld - 1e-5:
                        operInfo[operUsed, 'score'] += pi2_ALNS     #算子分数+pi2
                    elif self.x not in xObtained:
                        operInfo[operUsed, 'score'] += pi3_ALNS     #算子分数+pi3

                self.history['x'].append(self.x)
                self.history['obj'].append(self.obj)
                self.T = self.T * self.theta_SA

            # 每个segment结束，更新算子的权重
            for k in range(operNum_ALNS):
                if operInfo[k,'num'] == 0:
                    self.weightOper[k] = (1-rho_ALNS) * self.weightOper[k]
                else:
                    self.weightOper[k] = (1-rho_ALNS) * self.weightOper[k] + rho_ALNS * operInfo[k,'score'] / operInfo[k,'num']
            print('segment=', step, 'weight', self.weightOper, 'operInfo=', operInfo.values())

            # record operation info
            recordWeight.append(copy.deepcopy(self.weightOper))
            recordOperInfo.append(copy.deepcopy(operInfo))

            # 每个segment结束，更新operInfo为默认值0
            for key in operInfo.keys():
                operInfo[key] = 0


if __name__ == '__main__':
    # xInitial = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    xInitial = initialSolution.feasibleSol(sI, pJ, sJ, pT, sT, sT_End, sT_0, sR, sR_End, sR_1, pG, pD, pR0, LT_scale, LT_shape, sOmega_eval)
    pOmega_ALNS = 50

    objInitial = Mobj.objRAMP(xInitial, sI, sJ, pT, sT, sT_End, sT_0, sR, sR_End, sR_1, pCP, pCM, pCF, pG, pD, pR0, LT_scale, LT_shape, pOmega_ALNS)
    T0_SA = 0.03*objInitial/math.log(2)     # initial temperature in SA
    theta_SA = (math.log(2)/ math.log(1e5))**(1/1000)


    alns = ALNS(T0_SA, theta_SA, xInitial, objInitial, pOmega_ALNS)
    alns.run()

    objReal = Mobj.objRAMP(alns.xBest, sI, sJ, pT, sT, sT_End, sT_0, sR, sR_End, sR_1, pCP, pCM, pCF, pG, pD, pR0, LT_scale,
                 LT_shape, pOmega_eval)
    print('objBest = ', alns.objBest)
    print('objReal = ', objReal)













