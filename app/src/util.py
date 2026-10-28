import io

import pandas as pd
from pandas import DataFrame, Series


def read_file_from_streamlit(uploaded_file)->DataFrame:
    """Read a file from a given streamlit file."""
    if uploaded_file.size>200*1024*1024:
        raise ValueError("Too large file")
    file_extension=uploaded_file.name.split('.')[-1]

    if file_extension=='csv':
        return pd.read_csv(uploaded_file)
    elif file_extension=='json':
        return pd.read_json(uploaded_file)
    elif file_extension in ['xls','xlsx']:
        return pd.read_excel(io.BytesIO(uploaded_file.read()),engine='openpyxl')
    else:
        raise ValueError(f"Unsupported file format:{file_extension}")

def attribute_info(df)->tuple[list,str,str]:
    """返回列名，数据类型，前10条数据"""
    #列名,['age','name','sex',...]
    attributes:list[str]=df.columns.tolist()
    #列的数据类型,['int64','object','bool',...]
    dtypes_df:Series=df.dtypes
    types_info="\n".join([f"{index}:{dtype}" for index,dtype in dtypes_df.items()])
    head_info=df.head(10).to_csv()
    return attributes,types_info,head_info

def contain_null_attributes_info(df:DataFrame)->tuple[list,str,str]:
    """
    返回有缺失值的列名，描述信息(count,mean,std,50%)，数据类型
    :param df:
    :return:
    """
    #获取有缺失值的列名
    attributes=df.columns[df.isnull().any()].tolist()
    if not attributes:
        return [],"",""

    description_info:DataFrame=df[attributes].describe(percentiles=[.5])    #索引：count,std,mean,50%
    description_info:str=description_info.loc[['count','mean','50%','std']].round(2).to_csv()

    dtypes_df=df[attributes].dtypes
    types_info="\n".join([f"{index}:{dtype}" for index,dtype in dtypes_df.items()])
    return attributes,types_info,description_info

def separate_fill_null_list(fill_null_dict:dict[str,int])->tuple[list,list,list,list,list]:
    """
    返回五个列表，每个列表里的列名按照指定方式填充
    """
    mean_list=[key for key,value in fill_null_dict.items() if value==1]
    median_list=[key for key,value in fill_null_dict.items() if value==2]
    mode_list = [key for key, value in fill_null_dict.items() if value == 3]
    new_category_list = [key for key, value in fill_null_dict.items() if value == 4]
    interpolation_list = [key for key, value in fill_null_dict.items() if value == 5]
    return mean_list, median_list, mode_list, new_category_list, interpolation_list

def check_all_columns_numeric(df:DataFrame)->bool:
    """
    判断df是否所有的列都是数值类型
    :param df:
    :return: True or False
    """
    return df.select_dtypes(include=[int,float]).shape[1]==df.shape[1]

def non_numeric_columns_and_unique_number(df:DataFrame)->dict[str,int]:
    """
    返回列名以及每列唯一数值个数的对应关系
    :param df:
    :return:
    """
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns
    return {col: int(df[col].nunique()) for col in non_numeric_cols}

def separate_decode_list(decided_dict,Y_name)->tuple[list,list,list]:
    """
    将所有的非数值列分成三类，数值映射，ont-hot，舍弃
    """
    convert_int_cols=[key for key,value in decided_dict.items() if value==1]
    one_hot_cols=[key for key,value in decided_dict.items() if value==2]
    drop_cols=[key for key,value in decided_dict.items() if value==3]
    if Y_name and Y_name in one_hot_cols:
        one_hot_cols.remove(Y_name)
        convert_int_cols.append(Y_name)
    if Y_name and Y_name in drop_cols:
        drop_cols.remove(Y_name)
        convert_int_cols.append(Y_name)
    return convert_int_cols,one_hot_cols,drop_cols

def select_Y(df:DataFrame,Y_name):
    """
    拆分目标特征值和其他特征值
    :param df:
    :param Y_name:
    :return:
    """
    if Y_name in df.columns:
        X=df.drop(columns=[Y_name])
        Y=df[Y_name]
        return X,Y
    else:
        return -1,-1

def get_balance_info(df:DataFrame,Y_name):
    shape_info=df.shape
    description_info=df.describe().to_csv()
    balance_info=df[Y_name].value_counts().to_dict()
    return shape_info,description_info,balance_info

def get_data_overview(df:DataFrame):
    """
    获取DataFrame的形状、头部、轮廓和描述信息。
    :param df:
    :return:
    """
    shape_info = str(df.shape)
    head_info = df.head().to_csv()
    nunique_info = df.nunique().to_csv()
    description_info = df.describe(include='all').to_csv()
    return shape_info, head_info, nunique_info, description_info

def get_selected_models(model_dict:dict[str,int]):
    """字典转列表"""
    return list(model_dict.values())

def get_balance_method_name(number:int)->str:
    """获取平衡策略名称"""
    if number==1:
        return "RandomOverSampler"
    elif number==2:
        return "SMOTE"
    elif number==3:
        return "ADASYN"
    elif number==4:
        return "未处理"
    else:
        return "未知"

def get_model_name(number:int)->str:
    """获取模型名称"""
    if number==1:
        return "LogisticRegression"
    elif number==2:
        return "SVC"
    elif number==3:
        return "GaussianNB"
    elif number==4:
        return "RandomForestClassifier"
    elif number==5:
        return "AdaBoostClassifier"
    elif number==6:
        return "XGBClassifier"
    elif number==7:
        return "GradientBoostingClassifier"
    else:
        return "未知"