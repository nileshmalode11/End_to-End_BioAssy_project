import logging
from typing import Dict,Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

def train_test_split(X_pca_100,y_sample):
    """This function is to split the data into train test split
    input=X_pca_100 and y_sample
    outcome=X_train,X_test,y_train,y_test"""
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X_pca_100,y_sample,test_size=0.2)
    return X_train,X_test,y_train,y_test

def logreg(X_train,y_train):
    """This function is used to train your machine with logreg algo.
    input=X_train,y_train
    outcome=df_bio_logreg"""
    logreg=LogisticRegression()
    df_bio_logreg=logreg.fit(X_train,y_train)
    return df_bio_logreg

def logreg_pred(X_train,X_test,df_bio_logreg):
    """Function is to predict the outcomes
    input=X_train,X_test
    outcome=y_pred_train_log,y_pred_test_log"""
    y_pred_train_log=df_bio_logreg.predict(X_train)
    y_pred_test_log=df_bio_logreg.predict(X_test)
    print(y_pred_train_log)
    print(y_pred_test_log) 
    return y_pred_train_log,y_pred_test_log


def logreg_accuracy(y_train,y_pred_train_log,y_test,y_pred_test_log):
    """This function is to find accuracy_scote of logreg
    input=y_train,y_pred_train_log,y_test,y_pred_test_log
    outcome=logreg_train_accuracy,logreg_test_accuracy"""
    logreg_accuracy_train=accuracy_score(y_train,y_pred_train_log)
    logreg_accuracy_test=accuracy_score(y_test,y_pred_test_log)
    return logreg_accuracy_train,logreg_accuracy_test

def logreg_CReport(y_test,y_pred_test_log):
    """This function to find classification report
    input=y_test,y_pred_test_log
    outcome=logreg_classi_report"""
    logreg_Classi_report=classification_report(y_test,y_pred_test_log)
    print(logreg_Classi_report)
    return logreg_Classi_report

def logreg_CMatrix(y_test,y_pred_test_log):
    """function is to find confusion matrix
    input=y_test,y_pred_test_log
    outcome=logreg_confusion_matrix"""
    logreg_CMatrix=confusion_matrix(y_test,y_pred_test_log)
    return logreg_CMatrix

def decisionTree(X_train,y_train):
    """function is to use decisiontree algo.to train out machine
    input=X_train,y_train
    outcome=decisionTree"""
    dt=DecisionTreeClassifier()
    decisionTree=dt.fit(X_train,y_train)
    return decisionTree

def dt_pred(X_train,X_test,decisionTree):
    """function is to predict the output
    input=X_train,X_test
    outcome=dt_pred_train,dt_pred_test"""
    dt_pred_train=decisionTree.predict(X_train)
    dt_pred_test=decisionTree.predict(X_test)
    return dt_pred_train,dt_pred_test

def dt_score_CReport_CMatrix(y_train,y_test,dt_pred_train,dt_pred_test):
    """function to evaluate the performanace of decisiontree algo using accuracy_score,confusion_matrix,classification_report
    input=y_train,y_test,dt_pred_train,dt_pred_test
    outcome= """
    dt_score_train=accuracy_score(y_train,dt_pred_train)
    dt_score_test=accuracy_score(y_test,dt_pred_test)
    dt_CReport=classification_report(y_test,dt_pred_test)
    dt_CMatrix=confusion_matrix(y_test,dt_pred_test)
    return dt_score_test,dt_score_train,dt_CReport,dt_CMatrix

def RandomForest(X_train,y_train):
    """function is to train machine with rnadomforest algo
    input=X_train,y_train
    outcome=randomforest"""
    RF=RandomForestClassifier(n_estimators=100,max_depth=5,oob_score=True,random_state=0)
    RandomForest=RF.fit(X_train,y_train)
    return RandomForest

def RandomForest_pred(X_train,X_test,RandomForest):
    """function is to predict the outcome
    input=X_train,X_test
    outcome=rf_pred_train,rf_pred_test"""
    rf_pred_train=RandomForest.predict(X_train)
    rf_pred_test=RandomForest.predict(X_test)
    return rf_pred_test,rf_pred_train

def RF_score_CMatrix_CReport(y_train,y_test,rf_pred_train,rf_pred_test):
    """Function is to evaluate the performance of model
    input=y_train,y_test,rf_pred_train,rf_pred_test
    outcome="""
    RF_score_train=accuracy_score(y_train,rf_pred_train)
    RF_score_test=accuracy_score(y_test,rf_pred_test)
    rf_CMatrix=confusion_matrix(y_test,rf_pred_test)
    rf_CReport=classification_report(y_test,rf_pred_test)
    return RF_score_train,RF_score_test,rf_CMatrix,rf_CReport


def svm(X_train,y_train):
    """function is to train the model using svm algo
    input=X_train,y_train
    outcome=svm_model"""
    svm= SVC(C=.4, kernel='linear', gamma=2)
    svm_model=svm.fit(X_train, y_train)
    return svm_model

def svm_pred(X_train,X_test,svm_model):
    """function is to predict the outcome
    input=X_train,X_test
    outcome=svm_pred_train,svm_ored_test"""
    svm_pred_train=svm_model.predict(X_train)
    svm_pred_test=svm_model.predict(X_test)
    return svm_pred_train,svm_pred_test

def svm_score_CMatrix_CReport(y_train,y_test,svm_pred_train,svm_pred_test):
    """function is to evaluate performance of model using score,CReport and CMatrix
    input=y_train,y_test,svm_pred_train,svm_pred_test
    outcome=svm_score,svm_CMatrix,svm_CReport"""
    svm_score_train=accuracy_score(y_train,svm_pred_train)
    svm_score_test=accuracy_score(y_test,svm_pred_test)
    svm_CMatrix=confusion_matrix(y_test,svm_pred_test)
    svm_CReport=confusion_matrix(y_test,svm_pred_test)
    return svm_score_train,svm_score_test,svm_CMatrix,svm_CReport

def AdaBoost(X_train,y_train):
    """function is to train machine using adaboost algo.
    input=X_train,y_train
    outcome=adaboost_model"""
    AdaBoost=AdaBoostClassifier(random_state=20)
    AdaBoost_model=AdaBoost.fit(X_train,y_train)
    return AdaBoost_model

def AdaBoost_pred(X_train,X_test,AdaBoost_model):
    """function is predict the outcomes
    input=X_train,X_test,AdaBoost_model
    outcome=ABoost_pred_train,ABoost_pred_test"""
    Aboost_pred_train=AdaBoost_model.predict(X_train)
    Aboost_pred_test=AdaBoost_model.predict(X_test)
    return Aboost_pred_test,Aboost_pred_train

def Adaboost_score_CMatrix_CReport(y_train,y_test,Aboost_pred_train,Adoost_pred_test):
    """function is to evaluate performance of the model
    input=y_train,y_test,Aboost_pred_train,Adoost_pred_test
    outcome=Adaboost_score,adaboost_CMatrix,Adaboost_CReport"""
    AdaBoost_score_train=accuracy_score(y_train,Aboost_pred_train)
    Adaboost_score_test=accuracy_score(y_test,Adoost_pred_test)
    Adaboost_CMatrix=confusion_matrix(y_test,Adoost_pred_test)
    Adaboost_CReport=classification_report(y_test,Adoost_pred_test)
    return AdaBoost_score_train,Adaboost_score_test,Adaboost_CMatrix,Adaboost_CReport

