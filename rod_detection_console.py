import rodDetection
from sklearn.ensemble import RandomForestClassifier
from sklearn import  svm
lab_tr, fea_tr = rodDetection.loadDataSet('train.txt')
lab_t, fea_t = rodDetection.loadDataSet('test7_3.txt')

clf = RandomForestClassifier(n_estimators=500)
clf.fit(fea_tr, lab_tr)
print('RF Score is %f'%(clf.score(fea_t, lab_t)))

clf = svm.SVC()
clf.fit(fea_tr, lab_tr)
print('SVM score is %f'%(clf.score(fea_t, lab_t)))