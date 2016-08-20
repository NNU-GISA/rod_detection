import rodDetection
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import numpy as np
from random import *
import copy
from time import time
import matplotlib.pyplot as plt

lab_tr, fea_tr = rodDetection.loadDataSet('train.txt')
for i in range(len(lab_tr)):
    if lab_tr[i]==10:
        lab_tr[i] = 0
    else:
        pass
lab_t, fea_t = rodDetection.loadDataSet('test7_3.txt')

classLabels = np.array([0, 1, 3, 5])

# rodDetection.easyEnsemble(fea_tr, lab_tr, fea_t ,lab_t)
# rodDetection.balanceCascade(fea_tr, lab_tr, fea_t, lab_t)
fea_tr0, lab_tr0 = fea_tr[(lab_tr==0)], lab_tr[(lab_tr==0)]

rodDetection.nearMiss(fea_tr0, lab_tr0, fea_t, lab_t, kNN=18)

# m,n = fea_tr0.shape
# distance0 = np.zeros((m, m))
# t0 = time()
# for i in range(m):
#     for j in range(m):
#         for t in range(n):
#             distance0[i, j] += (fea_tr0[i, t]-fea_tr0[j ,t])**2
#     distance0[i, j] = distance0[i, j]**0.5
# print('time cost %f'%(time() - t0))
# minValue, maxValue = distance0.min(), distance0.max()
# span = maxValue - minValue
# boxNum = 300
# delta = span/boxNum
#
# xBox = np.zeros((boxNum,))
# for i in range(boxNum):
#     l = minValue + i*delta
#     h = minValue + (i+1)*delta
#     xBox[i] = (~((distance0 >= l) ^ (distance0 <= h))).sum()
#
# x = [minValue+i*delta for i in range(boxNum)]
# plt.bar(x,xBox)
# # plt.show()
# distance0List = [list(temp) for temp in distance0]
#
# m, n = fea_tr0.shape
# firstSeedInd = randint(0,m-1)
# SeedInd = [randint(0,m-1)]
# notSeedInd = [i for i in list(range(m)) if i not in SeedInd]
# kNN = 18
# while(len(SeedInd)<kNN):
#     #no check
#     nextSeedInd = notSeedInd[0]
#     for SI in SeedInd:
#         for nSI in notSeedInd:
#             if distance0[SI][nSI] > distance0[SI][nextSeedInd]:
#                 nextSeedInd = nSI
#
#     SeedInd.append(nextSeedInd)
#     notSeedInd = [i for i in list(range(m)) if i not in SeedInd]
# print('SeedInd', SeedInd)
#
# classDict = dict( [ (i, [SeedInd[i]] ) for i in range(kNN) ] )
# seekClassDict =dict( [(v[0],k) for k,v in classDict.items()] )
# for nSI in notSeedInd:
#     likeObjectInd = SeedInd[0]
#     for SI in SeedInd:
#         if distance0[nSI][SI] < distance0[nSI][likeObjectInd]:
#             likeObjectInd = SI
#     classDict[seekClassDict[likeObjectInd]].append(nSI)
# print(classDict)
#
# chosenSample0 = []
#
# for key in range(kNN):
#     rowList = classDict[key]
#     if len(rowList)==1 or len(rowList) == 2:
#         keepEle = rowList[0]
#     else:
#         keepEle = rowList[0]
#         for ele in rowList:
#             reservedList = copy.copy(rowList)
#             reservedList = reservedList.remove(ele)
#             if(rodDetection.blockDistance(ele, reservedList, fea_tr0) \
#                < rodDetection.blockDistance(keepEle, rowList[1:], fea_tr0)):
#                 keepEle = ele
#     chosenSample0.append(keepEle)
# print('chosenSample = ', chosenSample0)





# clf = RandomForestClassifier(n_estimators=500)
# fea_tr0, lab_tr0 = fea_tr[(lab_tr==0)], lab_tr[(lab_tr==0)]
# fea_tr1, lab_tr1 = fea_tr[(lab_tr==1)], lab_tr[(lab_tr==1)]
# fea_tr3, lab_tr3 = fea_tr[(lab_tr==3)], lab_tr[(lab_tr==3)]
# fea_tr5, lab_tr5 = fea_tr[(lab_tr==5)], lab_tr[(lab_tr==5)]
#
# fea_tr0_left = fea_tr0
# fea_tr3_lfet = fea_tr3
# while(len(fea_tr0)>=20 or len(fea_tr3)>=20):
#     print(len(fea_tr0) ,'or', len(fea_tr3),'=',(len(fea_tr0)>=18 or len(fea_tr3)>=18))
#     ind0 = rodDetection.genTRan(0, fea_tr0.shape[0], fea_tr1.shape[0])
#     ind0_left = [i for i in range(len(lab_tr0)) if i not in list(set(ind0))]
#     ind3 = rodDetection.genTRan(0, fea_tr3.shape[0], fea_tr1.shape[0])
#     ind3_left = [i for i in range(len(lab_tr3)) if i not in list(set(ind3))]
#
#
#     fea_tr_i = np.vstack((fea_tr0[ind0], fea_tr1, fea_tr3[ind3], fea_tr5))
#     lab_tr_i = np.vstack((lab_tr0[ind0].reshape(-1,1), lab_tr1.reshape(-1, 1),\
#                         lab_tr3[ind3].reshape(-1,1), lab_tr5.reshape(-1, 1)))
#
#     fea_tr0_t_left = fea_tr0[ind0_left]
#     fea_tr3_t_left = fea_tr3[ind3_left]
#
#     fea_tr_t_i = np.vstack((fea_tr0_t_left, fea_tr3_t_left))
#     lab_tr_t_i = np.vstack((lab_tr0[ind0_left].reshape(-1,1), lab_tr3[ind3_left].reshape(-1, 1)))
#
#     clf.fit(fea_tr_i, lab_tr_i.reshape((len(lab_tr_i), )))
#     predi = clf.predict(fea_tr_t_i)
#     indi = (predi != lab_tr_t_i.reshape((len(lab_tr_t_i), )))
#     fea_tr_t_i_left = fea_tr_t_i[indi]
#     lab_tr_t_i_left = lab_tr_t_i[indi]
#     # extrac 0
#     indi = (lab_tr_t_i_left == 0).reshape( (len(lab_tr_t_i_left), ) )
#     fea_tr0_t_left = fea_tr_t_i_left[indi]
#     lab_tr0_t_left = lab_tr_t_i_left[indi]
#     #extract 3
#     indi = (lab_tr_t_i_left == 3).reshape( (len(lab_tr_t_i_left), ))
#     fea_tr3_t_left = fea_tr_t_i_left[indi]
#     lab_tr3_t_left = lab_tr_t_i_left[indi]
#
#     # join the left and train data
#     fea_tr0 = np.vstack((fea_tr0[ind0], fea_tr0_t_left))
#     lab_tr0 = np.vstack((lab_tr0[ind0].reshape(-1,1), lab_tr0_t_left.reshape(-1,1)))
#
#     fea_tr3 = np.vstack((fea_tr3[ind3], fea_tr3_t_left))
#     lab_tr3 = np.vstack((lab_tr3[ind3].reshape(-1,1), lab_tr3_t_left.reshape(-1,1)))



#
# t = {0:0, 1:0, 3:0, 5:0}
# p = [copy.deepcopy(t) for i in range(len(lab_t))]
# for i in range(100):
#     ind0 = rodDetection.genTRan(0, fea_tr0.shape[0], fea_tr1.shape[0])
#
#     ind3 = rodDetection.genTRan(0, fea_tr3.shape[0], fea_tr1.shape[0])
#
#     fea_tr = np.vstack((fea_tr0[ind0], fea_tr1, fea_tr3[ind3], fea_tr5))
#     lab_tr = np.vstack((lab_tr0[ind0].reshape(-1,1), lab_tr1.reshape(-1, 1),\
#                         lab_tr3[ind3].reshape(-1,1), lab_tr5.reshape(-1, 1)))
#     clf = RandomForestClassifier(n_estimators=500)
#     clf.fit(fea_tr, lab_tr.reshape((len(lab_tr),)))
#     pred = clf.predict(fea_t)
#     for j in range(len(lab_t)):
#         p[j][pred[j]] += 1
#     # print('Under sampling result is:')
#     # rodDetection.classifyResult(classLabels, lab_t, clf.predict(fea_t))
#
# final_pred = []
# for j in range(len(lab_t)):
#     temp = sorted(p[j].items(), key=lambda d:d[1], reverse = True)
#     final_pred.append(temp[0][0])
# final_pred = np.array(final_pred)
# rodDetection.classifyResult(classLabels, lab_t, final_pred)