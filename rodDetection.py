import numpy as np
import random
import copy
from sklearn.ensemble import RandomForestClassifier
from random import *
def loadDataSet(filename):
    '''
    Extract the data from dataSet file
    :param filename: the name of the datafile
    :return: (labels, features)
    '''
    fr = open(filename)
    dataArr = []
    for line in fr.readlines():
        line = line.strip().split(' ')
        line = [float(line[i]) for i in range(len(line)) ]
        dataArr.append(line)
    dataArr = np.array(dataArr)
    lab = dataArr[:, 0]
    fea = dataArr[:,1:]

    return lab, fea

def classifyResult(cla,label,  pred):
    '''

    :param cla: classLables lab和pred中有多少类别标签(array)classLabels = np.array([0, 1, 3, 5])
    :param label: 测试集标签(array)
    :param pred: 预测标签(array)
    :return: 输出一个rel\pred 类似表格的东西
    '''
    print('rel\pre', end='  ')
    for l in cla:
        print(l, end='   ')
    print('')
    for l in cla:
        ind = (label == l)
        predl = pred[ind]
        print(l, end='-----')
        for j in cla:
            print('%4d' % (predl == j).sum(), end='')
        print('')


def genTRan(sta, end, t):
    '''
    generate t random bootstrap numbers for start to end
    产生t 个随机数（有放回） 从sta到end
    :param sta: the start of the sequence
    :param end: the end of the range(satr,end) sequence
    :param t:  the number of results generated
    :return: the random sequence ind (lsit)
    '''
    ind = []
    for i in range(t):
        ind.append(random.randint(sta, end)-1)
    return ind

def easyEnsemble(fea_tr, lab_tr, fea_t, lab_t):
    '''
    用easyEnsemble的方式来减弱下采样信息缺失的效用
    :param fea_tr:
    :param lab_tr:
    :param fea_t:
    :param lab_t:
    :return: 用classfiResult输出easyEnsemble方式的效果
    '''
    ens_clf = RandomForestClassifier(n_estimators=500, warm_start=True)

    classLabels = np.array([0, 1, 3, 5])
    fea_tr0, lab_tr0 = fea_tr[(lab_tr == 0)], lab_tr[(lab_tr == 0)]
    fea_tr1, lab_tr1 = fea_tr[(lab_tr == 1)], lab_tr[(lab_tr == 1)]
    fea_tr3, lab_tr3 = fea_tr[(lab_tr == 3)], lab_tr[(lab_tr == 3)]
    fea_tr5, lab_tr5 = fea_tr[(lab_tr == 5)], lab_tr[(lab_tr == 5)]

    t = {0: 0, 1: 0, 3: 0, 5: 0}
    p = [copy.deepcopy(t) for i in range(len(lab_t))]
    for i in range(100):
        ind0 = genTRan(0, fea_tr0.shape[0], fea_tr1.shape[0])

        ind3 = genTRan(0, fea_tr3.shape[0], fea_tr1.shape[0])

        fea_tr = np.vstack((fea_tr0[ind0], fea_tr1, fea_tr3[ind3], fea_tr5))
        lab_tr = np.vstack((lab_tr0[ind0].reshape(-1, 1), lab_tr1.reshape(-1, 1), \
                            lab_tr3[ind3].reshape(-1, 1), lab_tr5.reshape(-1, 1)))
        clf = RandomForestClassifier(n_estimators=500)
        clf.fit(fea_tr, lab_tr.reshape((len(lab_tr),)))

        ens_clf.fit(fea_tr, lab_tr.reshape((len(lab_tr),)))
        ens_clf.n_estimators += 500

        pred = clf.predict(fea_t)
        for j in range(len(lab_t)):
            p[j][pred[j]] += 1
            # print('Under sampling result is:')
            # rodDetection.classifyResult(classLabels, lab_t, clf.predict(fea_t))

    final_pred = []
    for j in range(len(lab_t)):
        temp = sorted(p[j].items(), key=lambda d: d[1], reverse=True)
        final_pred.append(temp[0][0])
    final_pred = np.array(final_pred)
    print('easyEnsemble classify result is :')
    classifyResult(classLabels, lab_t, final_pred)

    ens_pred = ens_clf.predict(fea_t)
    print('ens_pred (test with warm_statr = True) can be compared/'
          'with last result')
    classifyResult(classLabels, lab_t, ens_pred)

def balanceCascade(fea_tr, lab_tr, fea_t, lab_t):
    '''
    用balanceCascade的方法来减弱下采样信息缺失 用classifyResult来输出结果
    :param fea_tr:
    :param lab_tr:
    :param lab_t:
    :param fea_t:
    :return:
    '''
    classLabels = np.array([0, 1, 3, 5])

    clf = RandomForestClassifier(n_estimators=500)
    fea_tr0, lab_tr0 = fea_tr[(lab_tr == 0)], lab_tr[(lab_tr == 0)]
    fea_tr1, lab_tr1 = fea_tr[(lab_tr == 1)], lab_tr[(lab_tr == 1)]
    fea_tr3, lab_tr3 = fea_tr[(lab_tr == 3)], lab_tr[(lab_tr == 3)]
    fea_tr5, lab_tr5 = fea_tr[(lab_tr == 5)], lab_tr[(lab_tr == 5)]

    t = {0: 0, 1: 0, 3: 0, 5: 0}
    p = [copy.deepcopy(t) for i in range(len(lab_t))]

    while (len(fea_tr0) >= 20 or len(fea_tr3) >= 20):

        ind0 = genTRan(0, fea_tr0.shape[0], fea_tr1.shape[0])
        ind0_left = [i for i in range(len(lab_tr0)) if i not in list(set(ind0))]
        ind3 = genTRan(0, fea_tr3.shape[0], fea_tr1.shape[0])
        ind3_left = [i for i in range(len(lab_tr3)) if i not in list(set(ind3))]

        fea_tr_i = np.vstack((fea_tr0[ind0], fea_tr1, fea_tr3[ind3], fea_tr5))
        lab_tr_i = np.vstack((lab_tr0[ind0].reshape(-1, 1), lab_tr1.reshape(-1, 1), \
                              lab_tr3[ind3].reshape(-1, 1), lab_tr5.reshape(-1, 1)))

        fea_tr0_t_left = fea_tr0[ind0_left]
        fea_tr3_t_left = fea_tr3[ind3_left]

        fea_tr_t_i = np.vstack((fea_tr0_t_left, fea_tr3_t_left))
        lab_tr_t_i = np.vstack((lab_tr0[ind0_left].reshape(-1, 1), lab_tr3[ind3_left].reshape(-1, 1)))

        clf.fit(fea_tr_i, lab_tr_i.reshape((len(lab_tr_i),)))
        predi = clf.predict(fea_tr_t_i)
        indi = (predi != lab_tr_t_i.reshape((len(lab_tr_t_i),)))
        fea_tr_t_i_left = fea_tr_t_i[indi]
        lab_tr_t_i_left = lab_tr_t_i[indi]
        # extrac 0
        indi = (lab_tr_t_i_left == 0).reshape((len(lab_tr_t_i_left),))
        fea_tr0_t_left = fea_tr_t_i_left[indi]
        lab_tr0_t_left = lab_tr_t_i_left[indi]
        # extract 3
        indi = (lab_tr_t_i_left == 3).reshape((len(lab_tr_t_i_left),))
        fea_tr3_t_left = fea_tr_t_i_left[indi]
        lab_tr3_t_left = lab_tr_t_i_left[indi]

        # join the left and train data
        fea_tr0 = np.vstack((fea_tr0[ind0], fea_tr0_t_left))
        lab_tr0 = np.vstack((lab_tr0[ind0].reshape(-1, 1), lab_tr0_t_left.reshape(-1, 1)))

        fea_tr3 = np.vstack((fea_tr3[ind3], fea_tr3_t_left))
        lab_tr3 = np.vstack((lab_tr3[ind3].reshape(-1, 1), lab_tr3_t_left.reshape(-1, 1)))

        # Predcition part
        pred = clf.predict(fea_t)
        for j in range(len(lab_t)):
            p[j][pred[j]] += 1
    final_pred = []
    for j in range(len(lab_t)):
        temp = sorted(p[j].items(), key=lambda d: d[1], reverse=True)
        final_pred.append(temp[0][0])
    final_pred = np.array(final_pred)
    classifyResult(classLabels, lab_t, final_pred)

def blockDistance(ele, reservedList, block):
    '''
    返回第ele个样本到reservedList中样本的街区距离和
    :param ele:(int) 目标样本
    :param reservedList: (list) location样本的位置
    :param block: features[n_samples, m_variables]
    :return: 第ele个样本到reservedList中样本的街区距离和
    '''
    sumDistance = 0
    objectSample = block[ele]
    baseSample = block[reservedList]
    sumDistance = abs(baseSample-objectSample).sum()
    return sumDistance


def nearMiss(feature,label, fea_t, lab_t, kNN):
    '''
    利用KNN试图挑选那些最具代表性的大众样本，叫做NearMiss
    减弱下采样信息的缺失
    :param fea_ture: 传入特征[n_samples, n_features]
    :param kNN: 将样本分为kNN类
    :return: chosenSample0 (list) 在feature中具有代表性的样本的下标
    '''
    m,n = feature.shape
    distance0 = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            for t in range(n):
                distance0[i, j] += (feature[i, t]-feature[j ,t])**2
        distance0[i, j] = distance0[i, j]**0.5
    m, n = feature.shape
    SeedInd = [randint(0,m-1)]
    notSeedInd = [i for i in list(range(m)) if i not in SeedInd]
    while(len(SeedInd)<kNN):
        #no check
        nextSeedInd = notSeedInd[0]
        for SI in SeedInd:
            for nSI in notSeedInd:
                if distance0[SI][nSI] > distance0[SI][nextSeedInd]:
                    nextSeedInd = nSI

        SeedInd.append(nextSeedInd)
        notSeedInd = [i for i in list(range(m)) if i not in SeedInd]
    # print('SeedInd', SeedInd)

    classDict = dict( [ (i, [SeedInd[i]] ) for i in range(kNN) ] )
    seekClassDict =dict( [(v[0],k) for k,v in classDict.items()] )
    for nSI in notSeedInd:
        likeObjectInd = SeedInd[0]
        for SI in SeedInd:
            if distance0[nSI][SI] < distance0[nSI][likeObjectInd]:
                likeObjectInd = SI
        classDict[seekClassDict[likeObjectInd]].append(nSI)
    print(classDict)

    chosenSample0 = []

    for key in range(kNN):
        rowList = classDict[key]
        if len(rowList)==1 or len(rowList) == 2:
            keepEle = rowList[0]
        else:
            keepEle = rowList[0]
            for ele in rowList:
                reservedList = copy.copy(rowList)
                reservedList = reservedList.remove(ele)
                if(blockDistance(ele, reservedList, feature) \
                   < blockDistance(keepEle, rowList[1:], feature)):
                    keepEle = ele
        chosenSample0.append(keepEle)

    classLabels = np.array([0, 1, 3, 5])
    fea_tr, lab_tr = feature[chosenSample0], label[chosenSample0]
    clf = RandomForestClassifier(n_estimators=500)
    clf.fit(fea_tr, lab_tr)
    pred = clf.predict(fea_t)
    classifyResult(cla=classLabels, pred=pred, label=lab_t)

