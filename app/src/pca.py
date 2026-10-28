import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .preprocess import convert_to_integer


def decide_pca(df:DataFrame,cumulative_variance_threshold:float=0.95,min_dim_reduction_ratio:float=0.1):
    """
    判断是否需要PCA降维

    Return:
         perform_pca:是否降维
         n_components:要降到的维数
    """
    numeric_df=df.select_dtypes(include=[np.number])
    #标准化
    scalar=StandardScaler()
    scaled_data=scalar.fit_transform(numeric_df)
    #PCA降维训练
    pca=PCA()
    pca.fit(scaled_data)
    #计算累积方差
    #pca.explained_variance_ratio_=[0.72, 0.18, 0.07, 0.03]
    #cumulative_variance=[0.72,0.9,0.97,1.0]
    cumulative_variance=np.cumsum(pca.explained_variance_ratio_)
    #达到方差阈值所需最小的主成分个数，需要降维到的维数
    n_components=np.where(cumulative_variance>=cumulative_variance_threshold)[0][0]+1
    #减少维数的占比，为0.8说明减少了8成的维数
    dim_reduction_ratio=1-(n_components/df.shape[1])
    #不能小于0.1，减的太少就没必要减了
    perform_pca=dim_reduction_ratio>=min_dim_reduction_ratio
    return perform_pca,n_components

def perform_pca(df:DataFrame,n_components,Y_name):
    """
    进行PCA降维
    :param df:
    :param n_components:
    :param Y_name:
    :return:
    """
    numeric_df=df.select_dtypes(include=[np.number])
    if Y_name and Y_name in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[Y_name])
    #标准化
    scalar=StandardScaler()
    scaled_data=scalar.fit_transform(numeric_df)
    #PCA降维
    pca=PCA(n_components=n_components)
    principal_components:DataFrame=pca.fit_transform(scaled_data)#降维后的数据

    columns=[f'PC{i+1}' for i in range(n_components)]#主成分名称
    pca_df=pd.DataFrame(data=principal_components,columns=columns)

    if Y_name:
        pca_df[Y_name]=df[Y_name].to_numpy()
        pca_df,_=convert_to_integer(pca_df,convert_int_cols=[Y_name])
    return pca_df