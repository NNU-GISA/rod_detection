import numpy as np
import random
import copy
from sklearn.ensemble import RandomForestClassifier
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
    """

    :rtype: object
    """
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
    classifyResult(classLabels, lab_t, final_pred)