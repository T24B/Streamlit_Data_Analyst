from collections import Counter
import streamlit as st
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.metrics import roc_curve, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import io
from joblib import dump

def check_and_balance(X,Y,balance_threshold=0.5,method=1):
    """
    检查数据集是否不平衡，必要时使用RandomOverSampler、SMOTE或ADASYN进行过采样。
    :param X:
    :param Y:
    :param balance_threshold:
    :param method:
    :return:
    """
    try:
        #统计目标特征值里的每个类别出现的次数  {0:100,1:50,2:59}
        class_distribution=Counter(Y)
        min_class_samples=min(class_distribution.values())
        max_class_samples=max(class_distribution.values())
        #最少类别和最多类别比例大于阈值则说明不平衡
        is_imbalance=min_class_samples/max_class_samples>=balance_threshold

        if is_imbalance and method!=4:
            if method==1:
                oversampler=RandomOverSampler(random_state=0)
            elif method==2:
                oversampler=SMOTE(random_state=0)
            elif method==3:
                oversampler=ADASYN(random_state=0)
            X_resampled,Y_resampled=oversampler.fit_resample(X,Y)
            return X_resampled,Y_resampled
        else:
            return X,Y
    except Exception as e:
        st.error("目标属性可能是连续的，请检查数据类型")
        st.stop()

def split_data(X,Y,test_size=0.2,random_state=42,perform_pca=False):
    """
    拆分test,train
    :param X:
    :param Y:
    :param test_size:
    :param random_state:
    :param perform_pca:
    :return:
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    #没PCA过
    if not perform_pca:
        scaler=StandardScaler()
        X_train=scaler.fit_transform(X_train)
        X_test=scaler.transform(X_test)
    return X_train,X_test,Y_train,Y_test

def save_model(model):
    """
    使用joblib的dump函数将机器学习模型序列化为二进制格式，并存储在BytesIO缓冲区中
    :param model:
    :return:
    """
    with io.BytesIO() as buffer:
        #joblib写入二进制数据
        dump(model,buffer)
        return buffer.getvalue()

def fpr_and_tpr(model,X_test,Y_test):
    """
    计算分类模型的误报率FPR和真阳性率TPR。
    :param model:
    :param X_test:
    :param Y_test:
    :return:
    """
    Y_pred=model.predict_proba(X_test)[:,1]
    fpr,tpr,_=roc_curve(Y_test,Y_pred)
    return fpr,tpr

def calculate_f1_score(model,X_test,Y_test,binary_classification=True):
    Y_pred=model.predict(X_test)
    if binary_classification:
        f1=f1_score(Y_test,Y_pred,average='binary')
    else:
        f1=f1_score(Y_test,Y_pred,average='macro')
    return f1