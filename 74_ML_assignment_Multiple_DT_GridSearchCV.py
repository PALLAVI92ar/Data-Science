import pandas as pd
df=pd.read_csv("breast-cancer-wisconsin-data.csv")
df.shape
list(df)

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
df['diagnosis_encoded']=LE.fit_transform(df['diagnosis'])
df['diagnosis_encoded']

df.shape
list(df)

##########################################################

X=df.iloc[:,2:32]
X
list(X)

from sklearn.preprocessing import StandardScaler
X_scale=StandardScaler().fit_transform(X)
X_scale
X_scale.shape


Y=df['diagnosis_encoded']
Y
###############################################################

from sklearn.model_selection._split import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X_scale,Y,test_size=0.30,random_state=50,stratify=Y)

X_train.shape
X_test.shape
Y_train.shape
Y_test.shape

from sklearn.linear_model import LogisticRegression
lr1=LogisticRegression()
lr1.fit(X_train,Y_train)

Y_pred1=lr1.predict(X_test)
Y_pred1

from sklearn import metrics
cm=metrics.confusion_matrix(Y_test,Y_pred1)
cm

metrics.accuracy_score(Y_pred1,Y_test).round(3)#--->0.988

import matplotlib.pyplot as plt
plt.matshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.ylabel('predicted value')
plt.xlabel('Actual test value')
plt.show()


from sklearn.metrics import log_loss
l_loss1=log_loss(Y_pred1,Y_test)
print("Log loss -> ", l_loss1.round(3))#------>0.404

#################################################

from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV


alpha_val={'alpha': [0.001,0.002,0.003,0.004,0.005,0.009,0.01,0.02,0.03,0.04,0.05,0.06,0.1,0.2,0.3,0.4,0.5,1,2,3,5,10,20,50,100]}

ridge_cv=GridSearchCV(RidgeClassifier(),alpha_val,scoring='accuracy',cv=10)
ridge_cv


#r_m1=ridge_cv.fit(X_scale,Y)
#print(r_m1.best_score_)#--->0.961
#print(r_m1.best_params_)#--->10

r_m1=ridge_cv.fit(X_train,Y_train)
print(r_m1.best_score_)#--->0.959
print(r_m1.best_params_)#--->1

#############################################################

r_m2=RidgeClassifier(alpha=1)
r_m2.fit(X_train,Y_train)
r_m2.intercept_
r_m2.coef_

pred_train=r_m2.predict(X_train)
print ("Training log loss", log_loss(Y_train,pred_train).round(3))#-->1.041
print ("Training Accuracy", metrics.accuracy_score(Y_train,pred_train).round(3))#-->0.97

pred_test = r_m2.predict(X_test)
print ("Test log loss", log_loss(Y_test,pred_test).round(3))#-->1.414
print ("Test Accuracy", metrics.accuracy_score(Y_test,pred_test).round(3))#-->0.959

plt.figure(figsize = (12,6))
plt.axhline(0,color='red',linestyle='solid')
plt.plot(range(len(X.columns)), r_m2.coef_[0])

plt.show()

import numpy as np
coeff_estimator = pd.concat([pd.DataFrame(X.columns), pd.DataFrame(np.transpose(r_m2.coef_))] ,axis =1, ignore_index=True)
coeff_estimator

coeff_estimator.sort_values(by=1)
#############################################################

X1=df[df.columns[[28,21,7,29,22,6,10,20]]]
list(X1)

X_train,X_test,Y_train,Y_test=train_test_split(X1,Y, test_size=0.30, random_state=70, stratify=Y)

lr2=LogisticRegression()
lr2.fit(X_train,Y_train)

Y_pred2=lr2.predict(X_test)
Y_pred2

cm=metrics.confusion_matrix(Y_test,Y_pred2)
cm

metrics.accuracy_score(Y_pred2,Y_test).round(3)#-->0.953

l_loss2=log_loss(Y_pred2,Y_test)
print("Log loss -> ", l_loss2.round(2))#-->1.62

###############################################################
'''##ElasticNet won't works

from sklearn.linear_model import ElasticNet

alphaval={'alpha':[1],'l1_ratio':[0.001,0.002,0.01,0.02,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}
elastcv=GridSearchCV(ElasticNet(),alphaval,scoring='accuracy',cv=10)
elastcv

elastcv.fit(X_train,Y_train)

print(elastcv.best_score_)
print(elastcv.best_params_)

'''
##############################################################
##KNN##

X1=df[df.columns[[28,21,7,29,22,6,10,20]]]
list(X1)

X_train,X_test,Y_train,Y_test=train_test_split(X1,Y,test_size=0.30,random_state=70)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,Y_train)

Y_pred_knn=knn.predict(X_test)

cm1=metrics.confusion_matrix(Y_test,Y_pred_knn)
cm1

knn_acc = metrics.accuracy_score(Y_test,Y_pred_knn)
print("Accuracy for k=7",knn_acc)#-->0.935

###########################################################
from sklearn.model_selection import GridSearchCV

k=range(1,51)
P_val=[1,2]

hyperparameters=dict(n_neighbors =k, p=P_val)
clf=GridSearchCV(KNeighborsClassifier(),hyperparameters,cv=10)

knn1=clf.fit(X_train,Y_train)

knn1.cv_results_

print(knn1.best_score_) #--------------->93.4

print(knn1.best_params_)#-------------------->k=15 p=1


##############################
'''based on the results from logistic and KNN
we can conclude that logistic regression gets best accuracy score than KNN
logistic reg-accuracy score=95.3
KNN classifier -best score =93.4
'''
##########################################################################
##september 28th assignment
##Decision tree##
from sklearn.tree import DecisionTreeClassifier

DTC1=DecisionTreeClassifier(criterion='entropy')

DTC1.fit(X_train,Y_train)

print(f"Decision tree contains {DTC1.tree_.node_count} nodes with maximum depth = {DTC1.tree_.max_depth}")
##43-45 nodes ,,depth=9-10

Y_pred_dt=DTC1.predict(X_test)

cm2=metrics.confusion_matrix(Y_test,Y_pred_dt) 
cm2

acc2=metrics.accuracy_score(Y_test,Y_pred_dt) 
acc2#---->94.7% with 45 nodes, depth 10

depth_setting=range(1,12)
training_scores=[]
test_scores=[]

for x in depth_setting:
    DTC2=DecisionTreeClassifier(criterion='entropy',max_depth=x)
    DTC2.fit(X_train,Y_train)
    
    train_y_pred_dt1=DTC2.predict(X_train)
    test_y_pred_dt1=DTC2.predict(X_test)
    
    training_scores.append(metrics.accuracy_score(Y_train,train_y_pred_dt1))
    test_scores.append(metrics.accuracy_score(Y_test,test_y_pred_dt1))

print(pd.DataFrame(training_scores).round(2))
print(pd.DataFrame(test_scores).round(2))###-->at depth 4 ,,score=96.4%


#####################################

##set depth=4

DTC3=DecisionTreeClassifier(criterion='entropy',max_depth=4)

DTC3.fit(X_train,Y_train)

print(f"Decision tree contains {DTC3.tree_.node_count} nodes with maximum depth = {DTC3.tree_.max_depth}")

Y_pred_dt2=DTC3.predict(X_test)

cm3=metrics.confusion_matrix(Y_test,Y_pred_dt2) 
cm3

acc3=metrics.accuracy_score(Y_test,Y_pred_dt2) 
acc3##---> at depth 4--> nodes reduced to 21 and accuracy score=96.4%

'''
Decision tree-accuracy score=96.4%
logistic reg-accuracy score=95.3%
KNN classifier -best score =93.4% 
As of now among this 3 ,,best is decision tree because of with its best accuracy result
'''
##########################################
##Oct 1st assignment
# with multiple tree

from sklearn.tree import DecisionTreeClassifier

DTC1=DecisionTreeClassifier(criterion='entropy')

DTC1.fit(X_train,Y_train)

print(f"Decision tree contains {DTC1.tree_.node_count} nodes with maximum depth = {DTC1.tree_.max_depth}")
##43-45 nodes ,,depth=9-10

Y_pred_dt=DTC1.predict(X_test)

cm2=metrics.confusion_matrix(Y_test,Y_pred_dt) 
cm2

acc2=metrics.accuracy_score(Y_test,Y_pred_dt) 
acc2#---->94.7% with 45 nodes, depth 10

##apply gridsearchcv 

from sklearn.model_selection import GridSearchCV
levels={'max_depth': [1,2,3,4,5,6,7,8,9,10]}

grid=GridSearchCV(DTC1, cv=10, scoring='accuracy', param_grid=levels)
gridfit=grid.fit(X_train,Y_train)

gridfit.fit(X_test,Y_test)

print(gridfit.best_score_)##score=97%

gridfit.best_estimator_ ##depth=3
#################################

#apply bagging on base learner (DTC1)

from sklearn.ensemble import BaggingClassifier

bag=BaggingClassifier(base_estimator=DTC1,max_samples=0.6, n_estimators=500, random_state=10)
bag.fit(X_train,Y_train)

Y_pred_bag=bag.predict(X_test)
 
print(metrics.accuracy_score(Y_pred_bag,Y_test).round(3))#-->95.9%


###############################################################################
##test with different sample sizes
import numpy as np

setting_size=np.arange(0.1,1,0.1)
training_score=[]
test_score=[]

for x in setting_size:
   
    bag1=BaggingClassifier(base_estimator=DTC1,max_samples=x, n_estimators=500, random_state=10)
    bag1.fit(X_train,Y_train)
    
    Y_pred_train = bag1.predict(X_train)
    training_score.append(metrics.accuracy_score(Y_train,Y_pred_train).round(3))
    
    Y_pred_test = bag1.predict(X_test)
    test_score.append(metrics.accuracy_score(Y_test,Y_pred_test).round(3))

print(pd.DataFrame(training_score).round(2))
print(pd.DataFrame(test_score).round(2))

###############################################################################
##graphical visualization

import matplotlib.pyplot as plt
plt.plot(setting_size,training_score,label='Accuracy of the training data')
plt.plot(setting_size,test_score,label='Accuracy of the test data')
plt.ylabel('Accuracy score')
plt.xlabel('Percentage of Bagging samples')
plt.legend()

###############################################################################

# GridSearchCV method
from sklearn.model_selection import GridSearchCV
samples={'max_samples': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}

grid1= GridSearchCV(bag, cv=10, scoring='accuracy', param_grid=samples)
gridfit1=grid1.fit(X_train,Y_train)

gridfit1.fit(X_test,Y_test)

print(gridfit1.best_score_)##-->97.6%

gridfit1.best_estimator_

########################################################################
'''
compare to logistic,KNN,decision tree,,
for decision tree we got high accuracy score=97.6%
'''




