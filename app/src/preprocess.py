import pandas as pd
from pandas import DataFrame


def remove_rows_with_empty_target(df:DataFrame,Y_name):
    """
    删除目标特征值为空的行
    :param df:
    :param Y_name:
    :return:
    """
    return df.dropna(subset=[Y_name])

def convert_to_numeric(df:DataFrame,convert_int_cols:list,one_hot_cols:list,drop_cols:list):
    """
    将df的列按照相应办法进行转换，返回处理后的df，以及两种映射的实际映射关系
    """
    df,int_mapping=convert_to_integer(df,convert_int_cols)
    df,one_hot_mapping=convert_to_one_hot(df,one_hot_cols)
    df=df.drop(columns=drop_cols,errors='ignore')
    mappings={
        "integer_mappings":int_mapping,
        "ont_hot_mappings":one_hot_mapping
    }
    return df,mappings

def convert_to_integer(df:DataFrame,convert_int_cols:list)->tuple:
    """
    进行数值映射

    Returns:
        df    处理后的df
        mappings:{
            'department':{
                '信息部':0,
                '财政部':1,
                '人员部':2
            },
            'job':{
                '开发人员':0,
                '测试人员':1,
                '研究人员':2
            }
        }
    """
    mappings={}
    for col in convert_int_cols:
        if df[col].dtype=='object':
            #获取该列所有的唯一值  a,b,c
            unique_values=df[col].unique()
            #将0,1,2...与唯一值做一个对应 {0:a,1:b,2:c}
            int_to_value_map={i:value for i,value in enumerate(unique_values)}
            mappings[col]=int_to_value_map
            #映射关系反转一下  {a:0,b:1,c:2}
            value_to_int_map={v:k for k,v in int_to_value_map.items()}
            #转换
            df[col]=df[col].map(value_to_int_map)
    return df,mappings

def convert_to_one_hot(df:DataFrame,one_hot_cols:list)->tuple:
    """
    进行ont-hot编码
    """
    mappings={}
    df_modified=df.copy()

    for col in one_hot_cols:
        if df[col].dtype.name in ['object','category']:
            one_hot=pd.get_dummies(df[col],prefix=col)
            df_modified=pd.concat([df_modified,one_hot],axis=1)
            df_modified=df_modified.drop(columns=col)
            mappings[col]={i:col+'_'+str(i) for i in df[col].unique()}
    return df_modified,mappings

def remove_duplicates(df):
    """
    Remove duplicate rows from the DataFrame.
    """
    return df.drop_duplicates()