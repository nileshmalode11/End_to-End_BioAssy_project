import logging
import pandas as pd
import numpy as np
# def read_file():
#     """
#     Reads the csv file and returns a dataframe
#     """
    
#     Bioassay_comp_train_raw = pd.read_csv("C:\Users\Lenovo\Desktop\work\bio_assay_project\bio-assay-project\data\01_raw\AID746AID1284red_train.csv.zip")
#     Bioassay_comp_test_raw =pd.read_csv("C:\Users\Lenovo\Desktop\work\bio_assay_project\bio-assay-project\data\01_raw\AID746AID1284red_test.csv.zip")

#     return Bioassay_comp_train_raw,Bioassay_comp_test_raw

def concat(df746_1284_train,df746_1284_test):
    """This function is used to concat the train and test file of compound.
    input=Bioassay_comp_train_raw,Bioassay_comp_test_raw
    output=df_bio"""
    
    df_bio=pd.concat([df746_1284_train,df746_1284_test],axis=0,ignore_index=True)
    return df_bio

def df_bio_shape(df_bio):
    """This function is to show shape of df_bio
    input=df_bio
    output=df_bio_shape"""
    df_bio_shape=df_bio.shape
    print(df_bio_shape)
    return df_bio_shape

def df_bio_count(df_bio):
    """This function is to show value count of target variable to show data balance or not
    input=df_bio
    output=df_bio_count"""
    df_bio_count=df_bio["Outcome"].value_counts
    print(df_bio_count)
    return df_bio_count

def df_bio_split(df_bio):
    """This function is to split the data in X and y where X will store the predictor and y store responce variable
    input=df_bio
    outcome=X,y"""
    X=df_bio.drop(["Outcome"],axis=1)
    y=df_bio["Outcome"]
    print(y)
    return X,y

def df_bio_balance(X,y):
    """This function is to balance the data.Here we are using smote technique
    input=X,y
    outcome=df_bio_balance"""
    from imblearn.over_sampling import SMOTE
    sm=SMOTE()
    X_sample,y_sample=sm.fit_resample(X,y)
    print(y_sample)
    return X_sample,y_sample

def df_bio_scaling(X_sample):
    """This function is to scale the data.Here we are using standardscalar
    input=X_sample
    outcome=scaling_data"""
    from sklearn.preprocessing import StandardScaler
    scale=StandardScaler()
    scaling_data=scale.fit_transform(X_sample)
    print(scaling_data)
    return scaling_data

def df_bio_pca(scaling_data):
    """This function is for diamentationality reduction where n_componants=None 
    input=scaling_data
    outcome=X_pca"""
    from sklearn.decomposition import PCA
    pca=PCA(n_components=None)
    X_pca=pca.fit(scaling_data)
    X_pca=pca.transform(scaling_data)
    return X_pca

def df_bio_pca_100(scaling_data):
    """This function is for diamentationality reduction and here n_componants=100
    input=scaling_data
    outcome=X_pca_100"""
    from sklearn.decomposition import PCA
    pca=PCA(n_components=100)
    X_pca_100=pca.fit(scaling_data)
    X_pca_100=pca.transform(scaling_data)
    return X_pca_100





