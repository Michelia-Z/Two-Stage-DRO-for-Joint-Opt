# ex1.py 的 ‘写入/输出’ 文件

import xlwt
import readData
import numpy as np
import xlrd


# read data
filepath = 'data//'
pI, sI, pJ, sJ, pT, sT, sT_End, sT_0, pR, sR, sR_End, sR_1, pCP, pCM, pCF, pG, pD, pR0, LT_scale, LT_shape, pLT_mean, \
        pOmega_eval, sOmega_eval, probOmega, pSuppLower, pSuppUpper, pMu, pAbs, pGreaterMu = readData.readData(filepath)


# write InfoSet (ex1.py)
def printInfo(InfoSet):
    filename_info = filepath + 'Out-of-sample InfoSet.xls'
    f = xlwt.Workbook()     # 创建工作簿
    sheet0 = f.add_sheet(u'sheet1', cell_overwrite_ok = True)      # 创建sheet
    sheet0.write(0, 0, 'criteria')
    sheet0.write(0, 1, 'x')
    sheet0.write(0, 2, 'obj')
    sheet0.write(0, 3, 'obj-firstStage')
    sheet0.write(0, 4, 'min')
    sheet0.write(0, 5, 'max')
    sheet0.write(0, 6, 'aver')
    sheet0.write(0, 7, 'dec')
    sheet0.write(0, 8, 'qua')
    sheet0.write(0, 9, 'std')
    sheet0.write(0, 10, 'OBJ_OOS')
    sheet0.write(0, 11, 'infeasible')

    sheet0.write(1, 0, 'WC/BC/EM')
    sheet0.write(1, 1, 'optimal sol for RAMP')
    sheet0.write(1, 2, 'optimal obj for RAMP')
    sheet0.write(1, 3, 'first-stage obj of x')
    sheet0.write(1, 4, 'min(second-stage obj)')
    sheet0.write(1, 5, 'max(second-stage obj)')
    sheet0.write(1, 6, 'mean(second-stage obj)')
    sheet0.write(0, 7, 'dec(second-stage obj)')
    sheet0.write(0, 8, 'qua(second-stage obj)')
    sheet0.write(1, 9, 'std(second-stage obj)')
    sheet0.write(1, 10, 'total obj in out-of-sample')
    sheet0.write(1, 11, 'infeasible of Maintenance Problem(sample)')

    for k in range(len(InfoSet)):
        info = InfoSet[k]
        nrow = k+2
        sheet0.write(nrow, 0, info['criteria'])
        sheet0.write(nrow, 1, str(info['x']))
        sheet0.write(nrow, 2, info['obj'])
        sheet0.write(nrow, 3, info['obj-firstStage'])
        sheet0.write(nrow, 4, info['min'])
        sheet0.write(nrow, 5, info['max'])
        sheet0.write(nrow, 6, info['aver'])
        sheet0.write(nrow, 7, info['dec'])
        sheet0.write(nrow, 8, info['qua'])
        sheet0.write(nrow, 9, info['std'])
        sheet0.write(nrow, 10, str(info['OBJ_OOS']))
        sheet0.write(nrow, 11, info['infeasible'])
    # 保存文件
    f.save(filename_info)


# write xBest, objBest (ex1.py)
def printXObj(criteria, xBest, objBest):
    filename_xobj = filepath + criteria + ' optimal x+obj.xls'
    f = xlwt.Workbook()     # 创建工作簿
    sheet0 = f.add_sheet(u'sheet1', cell_overwrite_ok = True)      # 创建sheet
    sheet0.write(0, 0, 'xBest')
    sheet0.write(0, 1, 'objBest')
    sheet0.write(1, 0, str(xBest))
    sheet0.write(1, 1, objBest)
    # 保存文件
    f.save(filename_xobj)


# read xBest, objBest (ex1.py)
def readXObj(criteria):
    filename_xobj = filepath + criteria + ' optimal x+obj.xls'
    data = xlrd.open_workbook(filename_xobj)
    table = data.sheet_by_index(0)

    objBest = table.row(1)[1].value
    xBest_str = table.row(1)[0].value
    temp = xBest_str[1:-1].split(',')
    xBest = [int(a) for a in temp]

    return xBest, objBest

# write xBestSet, objBestSet (ex - RAP+M vs RAMP.py)
def printXObjSet(filename, xBestSet, objBestSet):
    filename_ = filepath + filename
    f = xlwt.Workbook()     # 创建工作簿
    sheet0 = f.add_sheet(u'sheet1', cell_overwrite_ok = True)      # 创建sheet

    sheet0.write(0, 0, 'model')
    sheet0.write(0, 1, 'xBest')
    sheet0.write(0, 2, 'objBest')
    sheet0.write(1, 0, 'RAP+M')
    sheet0.write(2, 0, 'RAMP')

    sheet0.write(1, 1, str(xBestSet[0]))
    sheet0.write(1, 2, objBestSet[0])
    sheet0.write(2, 1, str(xBestSet[1]))
    sheet0.write(2, 2, objBestSet[1])
    # 保存文件
    f.save(filename_)

