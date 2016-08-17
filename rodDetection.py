import numpy as np

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
        dataArr.append(line)
    dataArr = np.array(dataArr)
    lab = dataArr[:, 0]
    fea = dataArr[:,1:]

    return lab, fea