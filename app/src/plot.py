import nltk
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import streamlit as st
from pandas import DataFrame
from plotly import graph_objects
from plotly.graph_objs import Figure
from sklearn.metrics import confusion_matrix
import seaborn as sns
import plotly.express
from wordcloud import WordCloud

@st.cache_data#缓存函数返回的数据
def correlation_matrix_plotly(df):
    """
    使用Plotly绘制所有属性的相关热图。
     Args:
         df:m*n,n个属性
    """
    corr_matrix:DataFrame=df.corr()#相关系数矩阵n*n,对角线为1
    labels=corr_matrix.columns
    text=[[f'{corr_matrix.iloc[i,j]:.2f}' for j in range(len(labels))] for i in range(len(labels))]
    fig:Figure=graph_objects.Figure(
        data=graph_objects.Heatmap(
            x=labels,
            y=labels,
            z=corr_matrix.values,
            text=text,
            hoverinfo='text',
            colorscale='Viridis',#颜色映射方案
            colorbar=dict(title='Correlation')#颜色映射条的配置
        )
    )
    fig.update_layout(
        title='属性之间的相关性矩阵',
        xaxis=dict(tickmode='linear'),
        yaxis=dict(tickmode='linear'),
        width=800,
        height=700,
        font=dict(size=10)
    )
    return fig

def roc(model_name,fpr,tpr):
    """绘制ROC曲线"""
    with plt.style.context('ggplot'):
        fig=plt.figure()
        plt.plot([0,1],[0,1],'k--')#从(0,0)到(1,1)的一条黑色虚线
        plt.plot(fpr,tpr,label=model_name)

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC-{model_name}')
        plt.legend(loc='best')
        return fig

def confusion_metrix(model_name,model,X_test,Y_test):
    """
    绘制混淆矩阵
    :param model_name:
    :param model:
    :param X_test:
    :param Y_test:
    :return:
    """
    Y_pred=model.predict(X_test)
    #计算混淆矩阵
    matrix=confusion_matrix(Y_test,Y_pred)
    plt.figure(figsize=(10,7))
    #画热力图
    sns_heatmap=sns.heatmap(matrix,annot=True,cmap='Blues',fmt='g',annot_kws={'size':20})
    plt.title(f"Confusion Matrix for {model_name}",fontsize=20)
    plt.xlabel("Predicted labels",fontsize=16)
    plt.ylabel("True labels",fontsize=16)
    return sns_heatmap.figure

def count_Y(df,att):
    """单个属性分布的环形图。"""
    if att in df.columns and df[att].nunique()>=1:
        value_counts=df[att].value_counts()
        fig=plotly.express.pie(
            names=value_counts.index,
            values=value_counts.values,
            title=f"{att}的分布",
            hole=0.5,
            color_discrete_sequence=plotly.express.colors.sequential.Cividis_r
        )
        return fig
    return None

def violin_plot(df,atts):
    """多属性的小提琴图"""
    #多属性但是有非数值属性
    if len(atts)>1 and not all(df[atts].dtypes.apply(lambda x:np.issubdtype(x,np.number))):
        return -1
    fig=plotly.express.violin(df,y=atts,title=f"{','.join(atts)}的小提琴图",color_discrete_sequence=plotly.express.colors.sequential.Cividis_r)
    return fig

def distribution_histogram(df, att):
    """单属性的分布直方图"""
    #object or category
    if df[att].dtype=='object' or isinstance(df[att].dtype, pd.CategoricalDtype):
        codes,uniques=pd.factorize(df[att])  #返回编码后的整数数组以及唯一值数组
        temp_df=pd.DataFrame({att:codes})
        fig,ax=plt.subplots(figsize=(8,6))
        sns.histplot(temp_df,ax=ax,discrete=True,color='#e17160')
        ax.set_xticks(range(len(uniques)))
        ax.set_xticklabels(uniques,rotation=45,ha='right')
    else:
        fig,ax=plt.subplots(figsize=(6,4))  #返回画布和子图
        sns.histplot(df[att],ax=ax,color='#e17160')    #seaborn将直方图画到子图上
    ax.set_title(f"{att} distribution")
    return fig

def box_plot(df,atts):
    """多属性箱线图"""
    if len(atts)>1 and not all(df[atts].dtypes.apply(lambda x:np.issubdtype(x,np.number))):
        return -1
    fig=plotly.express.box(df,y=atts,title=f"{','.join(atts)}箱线图")
    return fig

def density_plot(df,att):
    """单属性密度图"""
    fig=plotly.express.density_contour(df,x=att,y=att,title=f"{att} 密度图")
    return fig

def strip_plot(df,atts):
    """多属性条形图"""
    if len(atts)>1 and not all(df[atts].dtypes.apply(lambda x:np.issubdtype(x,np.number))):
        return -1
    fig=plotly.express.strip(df,y=atts,title=f"{','.join(atts)}条形图")
    return fig

def distribution_boxplot(df,att):
    """单属性分布箱线图"""
    if df[att].dtype=='object' or isinstance(df[att].dtype, pd.CategoricalDtype):
        return -1
    fig,ax=plt.subplots(figsize=(8,6))
    sns.boxenplot(data=df[att],palette=['#32936f','#26a96c','#2bc016'])
    ax.set_title(f"{att} distribution boxplot")
    return fig

def multi_plot_heatmap(df,atts):
    """多属性热力图"""
    if not all(df[atts].dtypes.apply(lambda x:np.issubdtype(x,np.number))):
        return -1
    sns.set_theme()
    plt.figure(figsize=(10,8))
    sns.heatmap(df[atts].corr(),annot=True,cmap='viridis')
    plt.title('关系型热力图')
    return plt.gcf()

def multi_plot_line(df,atts):
    """多属性线性图"""
    if not all(df[atts].dtypes.apply(lambda x:np.issubdtype(x,np.number))):
        return -1
    if len(atts)>=2:
        plt.figure(figsize=(10,6))
        colors=plt.cm.viridis(np.linspace(0,1,len(atts)))
        for i,att in enumerate(atts):
            plt.plot(df.index,df[att],color=colors[i],label=att,marker='',linewidth=2)
        plt.legend()
        plt.xlabel(atts[0])
        plt.ylabel(atts[1])
        plt.title(f'{atts[0]} vs {atts[1]} line plot')
        return plt.gcf()
    else:
        return -2

def multi_plot_scatter(df,atts):
    if len(atts)<=1:
        return -1
    plt.figure(figsize=(10,6))
    if df[atts[0]].dtype not in [np.float64,np.int64]:
        x,x_labels=pd.factorize(df[atts[0]])
        plt.xticks(ticks=np.arange(len(x_labels)),labels=x_labels,rotation=45)
    else:
        x=df[atts[0]]

    if df[atts[1]].dtype not in [np.float64,np.int64]:
        y,y_labels=pd.factorize(df[atts[1]])
        plt.xticks(ticks=np.arange(len(y_labels)),labels=y_labels)
    else:
        y=df[atts[1]]

    plt.scatter(x,y,c=np.linspace(0,1,len(df)),cmap='viridis')
    plt.colorbar()
    plt.xlabel(atts[0])
    plt.ylabel(atts[1])
    plt.title(f"{atts[0]} vs {atts[1]} scatter plot")
    return plt.gcf()

@st.cache_data
def scatter_3d(df,x,y,z):
    try:
        return plotly.express.scatter_3d(df,x=x,y=y,z=z,color=z,color_continuous_scale=plotly.express.colors.sequential.Viridis)
    except:
        return -1

@st.cache_data
def word_cloud_plot(text):
    try:
        words=nltk.regexp_tokenize(text,pattern='\w+')
        text_dict=nltk.FreqDist([w for w in words])
        wordcloud=WordCloud(width=1200,height=600,background_color='white').generate_from_frequencies(text_dict)
        fig,ax=plt.subplots(figsize=(10,7.5))
        ax.imshow(wordcloud,interpolation='bilinear')
        ax.axis("off")
        return fig
    except:
        return -1

@st.cache_data
def list_all(df,max_plots=16):
    num_plots=min(len(df.columns),max_plots)
    nrows=int(np.ceil(num_plots/4))
    ncols=min(num_plots,4)
    plt.style.use('ggplot')
    fig,axes=plt.subplots(nrows=nrows,ncols=ncols,figsize=(4*ncols,4*nrows))
    fig.suptitle('attributes distribution',fontsize=20)

    axes=axes.flatten()

    for i, ax in enumerate(axes[:num_plots]):
        col = df.columns[i]
        data = df[col].dropna()
        if pd.api.types.is_numeric_dtype(data):
            ax.hist(data, bins=20, color='steelblue')
        else:
            data.value_counts().head(10).plot(kind='bar', ax=ax)
        ax.set_title(col)
        ax.set_xlabel('')


    for ax in axes[num_plots:]:
        ax.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig