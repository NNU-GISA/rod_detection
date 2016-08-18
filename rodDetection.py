import numpy as np
import random
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

