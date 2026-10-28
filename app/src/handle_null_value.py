from pandas import Series, DataFrame


def contains_missing_value(df):
    """
    判断是否有缺失值

    Returns:
        True or False
    """
    return df.isnull().values.any()

def remove_high_null(df:DataFrame,threshold_row=0.5,threshold_col=0.7):
    """
    删除缺失值占比大于阈值的行和列
    :param df:
    :param threshold_row:
    :param threshold_col:
    :return:
    """
    #计算每列缺失值占比
    null_prop_col:Series=df.isnull().mean(axis=0)    #索引是列名，值是缺失值占比
    cols_to_drop=null_prop_col[null_prop_col>threshold_col].index    #获取缺失值占比超过阈值的列名
    df_cleaned=df.drop(columns=cols_to_drop)

    noll_prop_row=df_cleaned.isnull().mean(axis=1)
    rows_to_drop=noll_prop_row[noll_prop_row>threshold_row].index
    df_cleaned=df_cleaned.drop(index=rows_to_drop)

    return df_cleaned

def fill_null_values(df:DataFrame,mean_list,median_list,mode_list,new_category_list,interpolation_list):
    """
    根据传入的列名和插入方式的对照列表进行缺失值填充
    :param df:
    :param mean_list:
    :param median_list:
    :param mode_list:
    :param new_category_list:
    :param interpolation_list:
    :return:
    """
    if mean_list:
        df=fill_with_mean(df,mean_list)
    if median_list:
        df = fill_with_median(df, median_list)
    if mode_list:
        df = fill_with_mode(df, mode_list)
    if new_category_list:
        df = fill_with_NaN(df, new_category_list)
    if interpolation_list:
        df = fill_with_interpolation(df, interpolation_list)
    return df

def fill_with_mean(df:DataFrame,mean_list):
    for attr in mean_list:
        if attr in df.columns:
            df[attr]=df[attr].fillna(df[attr].mean())
    return df

def fill_with_median(df:DataFrame,median_list):
    for attr in median_list:
        if attr in df.columns:
            df[attr]=df[attr].fillna(df[attr].median())
    return df

def fill_with_mode(df:DataFrame,mode_list):
    for attr in mode_list:
        if attr in df.columns:
            mode_value=df[attr].mode()[0] if not df[attr].mode().empty else None
            if mode_value:
                df[attr]=df[attr].fillna(mode_value)
    return df

def fill_with_NaN(df:DataFrame,new_category_list):
    for attr in new_category_list:
        if attr in df.columns:
            df[attr]=df[attr].fillna('NaN')
    return df

def fill_with_interpolation(df:DataFrame,interpolation_list,method='linear'):
    for attr in interpolation_list:
        if attr in df.columns:
            df[attr]=df[attr].interpolate(method=method)
    return df
