import streamlit as st
from pandas import DataFrame
from sklearn.metrics import auc

from src.predictive_model import train_selected_model
from src.model_service import check_and_balance, split_data, save_model, fpr_and_tpr, calculate_f1_score
from src.pca import decide_pca, perform_pca
from src.plot import correlation_matrix_plotly, roc, confusion_metrix
from src.preprocess import remove_rows_with_empty_target, convert_to_numeric, remove_duplicates
from src.handle_null_value import contains_missing_value, remove_high_null, fill_null_values
from src.llm_service import decide_target_attribute, decide_fill_null, decide_encode_type, decide_test_ratio, \
    decide_balance, decide_model
from src.util import attribute_info, contain_null_attributes_info, separate_fill_null_list, check_all_columns_numeric, \
    separate_decode_list, non_numeric_columns_and_unique_number, select_Y, get_balance_info, get_data_overview, \
    get_selected_models, get_balance_method_name, get_model_name

"""
data_origin:用户上传的文件(1)

target_selected:
    False:还未选择目标特征值
    True:选择了目标特征值
target_Y:
    None:页面第一次加载,AI还没分析
    有值：AI选择的目标特征值
selected_Y:最终的目标特征值

contain_null:
    True:有缺失值
    False:无缺失值
filled_df:处理完缺失值后的文件(2)

df_cleaned:删除重复行后的文件(3)

all_numeric:
    True:data_origin所有列都是数值类型
    False:
encoded_df:编码后的文件(4)

to_perform_pca:
    True or False:是否PCA
df_pca:pca后的文件(5)

is_binary:
    True:二分类
    False:其他
test_percentage:
    10,20:测试集占比
start_training:
    True or False:是否已经开始训练
model_trained:

balance_data:
    True or False:是否需要平衡数据
balance_method:平衡数据的方法1,2,3,4

data_splitted:
    True or False:是否拆分了test和train
X_train,X_test,Y_train,Y_test:

decided_model:
    True or False:决定训练模型没有
all_set:
model_list:[1,3,4]
model1_name,model2_name,model3_name:[SVC,...,...]
model1,model2,model3:模型主体
downloadable_model1,downloadable_model2,downloadable_model3:供下载的模型主体
fpr1,tpr1,fpr2,tpr2,fpr3,tpr3:
has_been_set:
"""
def confirm_target(value):
    st.session_state.selected_Y = value
    st.session_state.target_selected = True

def start_train_model():
    st.session_state.start_training=True

def prediction_model_pipeline(DF_uploaded:DataFrame,API_KEY,MODEL):
    st.divider()

    #1、显示数据统计
    st.subheader('Data Overview')
    #存入数据源
    if 'data_origin' not in st.session_state:
        st.session_state.data_origin=DF_uploaded
    #数据统计，只统计数值列，个数，平均值，方差，分位数...
    st.dataframe(st.session_state.data_origin.describe(),width=1200)
    #所有列名
    attributes:list[str]=st.session_state.data_origin.columns.tolist()

    #2、选取目标特征值
    if 'target_selected' not in st.session_state:
        st.session_state.target_selected=False
    st.subheader('Target Variable')
    #页面第一次加载,AI分析,第二次target_Y就有值了，不会进入
    if 'target_Y' not in st.session_state:
        #显示分析页面
        with st.spinner('AI正在分析数据...'):
            #获取列名，数据类型，前10条数据
            attributes_for_target,types_info_for_target,head_info_for_target=attribute_info(st.session_state.data_origin)
            #获取目标列名，目标属性
            st.session_state.target_Y=decide_target_attribute(attributes_for_target,types_info_for_target,head_info_for_target,MODEL,API_KEY)
        #AI找到了
        if st.session_state.target_Y!=-1:
            st.session_state.selected_Y=st.session_state.target_Y
            st.session_state.target_selected=True
    #找到了目标特征值
    if st.session_state.target_selected:
        #AI没找到，所有只能是手动指定的了
        if st.session_state.target_Y == -1:
            st.info("目标特征值由你手动指定")
            st.write(f"目标特征值: :green[**{st.session_state.selected_Y}**]")
        else:
            st.info("目标特征值由AI指定")
            st.write(f"目标特征值: :green[**{st.session_state.selected_Y}**]")
    #AI没找到，自己手动指定
    else:
        st.info("AI找不出目标特征值。请手动选择")
        target_col1, target_col2 = st.columns([9, 1])
        with target_col1:
            st.session_state.selected_Y = st.selectbox(label='选择要预测的目标特征值:',options=attributes,index=len(attributes) - 1,label_visibility='collapsed')
        with target_col2:
            st.button('Confirm',type='primary',on_click=confirm_target,args=(st.session_state.selected_Y,))

    #3、处理并填补缺失值
    #找到了目标特征值
    if st.session_state.target_selected:
        st.subheader("Handle and Impute Missing Values")
        if "contain_null" not in st.session_state:
            st.session_state.contain_null=contains_missing_value(st.session_state.data_origin)

        #缺失值还没处理过
        if 'filled_df' not in st.session_state:
            #需要处理
            if st.session_state.contain_null:
                with st.status(label="处理缺失值过程中...",expanded=True) as status:
                    st.write("过滤掉高频缺失的行和列...")
                    filled_df=remove_high_null(DF_uploaded)
                    st.write("删除目标特征值为空的行...")
                    filled_df=remove_rows_with_empty_target(filled_df,st.session_state.selected_Y)

                    st.write("模型分析中...")
                    #分析出每一列应该用哪种填充方式：插值填充，众数填充，均值填充等
                    attributes,types_info,description_info=contain_null_attributes_info(filled_df)
                    fill_result_dict:dict[str,int]=decide_fill_null(attributes,types_info,description_info,MODEL,API_KEY)
                    st.write(f"{fill_result_dict}\n(1:均值填充\n2:中位数填充\n3:众数填充\n4:新类别填充\n5:插值填充)")

                    st.write("处理缺失值...")
                    mean_list,median_list,model_list,new_category_list,interpolation_list=separate_fill_null_list(fill_result_dict)
                    filled_df=fill_null_values(filled_df,mean_list,median_list,model_list,new_category_list,interpolation_list)

                    st.session_state.filled_df=filled_df
                    status.update(label="缺失值处理完毕",state="complete",expanded=False)
                    st.rerun()
            #不需要处理
            else:
                st.session_state.filled_df=DF_uploaded
                st.success("侦测到无缺失值，跳过处理过程")
                st.rerun()
        #缺失值处理过了
        else:
            st.success("缺失值处理完成")
            if st.session_state.contain_null:
                st.download_button(label="下载处理完毕的文件",data=st.session_state.filled_df.to_csv(index=False).encode('utf-8'),file_name="imputed_missing_values.csv",mime='text/csv')

    #4、删除重复行
    if st.session_state.target_selected:
        st.subheader('删除重复实体')
        if 'df_cleaned' not in st.session_state:
            st.session_state.df_cleaned = remove_duplicates(st.session_state.filled_df)
        st.success(f"删除重复实体完成，共删除 {len(st.session_state.filled_df) - len(st.session_state.df_cleaned)} 行")

    #5、字符型数据进行编码
    st.subheader("Process Data Encoding")
    if 'all_numeric' not in st.session_state:
        st.session_state.all_numeric=check_all_columns_numeric(st.session_state.df_cleaned)
    if 'encoded_df' not in st.session_state:
        #不全是数值类型
        if not st.session_state.all_numeric:
            with st.status("使用**numeric mapping**和**ont-hot**编码非数值数据",expanded=True) as status:
                #获取非数值列以及唯一值个数
                attributes_unique_info=non_numeric_columns_and_unique_number(st.session_state.df_cleaned)
                print(f"attributes_unique_info: {attributes_unique_info}")
                st.write("模型分析中...")
                #分析每列要进行编码的方式(数值映射，ont-hot,舍弃)
                encode_result_dict=decide_encode_type(attributes_unique_info,MODEL,API_KEY)
                print(f"选择的目标值{st.session_state.selected_Y}")
                print(f"AI对非数值列做出的编码方式判断: {encode_result_dict}")

                st.write("编码数据...")
                convert_int_cols,one_hot_cols,drop_cols=separate_decode_list(encode_result_dict,st.session_state.selected_Y)
                print(convert_int_cols,one_hot_cols,drop_cols)
                encoded_df,mappings=convert_to_numeric(st.session_state.df_cleaned,convert_int_cols,one_hot_cols,drop_cols)

                st.session_state.encoded_df=encoded_df
                status.update(label="数据编码完成",state="complete",expanded=False)
            st.download_button(label="下载编码后的数据",data=st.session_state.encoded_df.to_csv(index=False).encode('utf-8'),file_name="encoded_df.csv",mime='text/csv')
            st.rerun()
        else:
            st.session_state.encoded_df=st.session_state.df_cleaned
            st.success("所有的列都是数值型，跳过处理过程")
    else:
        st.success("数据已经成功使用numeric mapping和ont-hot编码")
        print(f"编码后的列数:{st.session_state.encoded_df.shape[1]}")
        if not st.session_state.all_numeric:
            st.download_button(label="下载编码后的数据",data=st.session_state.encoded_df.to_csv(index=False).encode('utf-8'),file_name="encoded_df.csv",mime='text/csv')

    #6、属性之间的相关性
    st.subheader("属性之间的相关性")
    st.plotly_chart(correlation_matrix_plotly(st.session_state.encoded_df))

    #7、PCA
    st.subheader('主成分分析')
    st.write('决定是否PCA中...')
    if 'df_pca' not in st.session_state:
        to_perform_pca,n_components=decide_pca(st.session_state.encoded_df.drop(columns=[st.session_state.selected_Y]))
        print(f"n_components: {n_components}")
        if 'to_perform_pca' not in st.session_state:
            st.session_state.to_perform_pca=to_perform_pca
        #需要PCA
        if st.session_state.to_perform_pca:
            st.session_state.df_pca=perform_pca(st.session_state.encoded_df,n_components,st.session_state.selected_Y)
        #不需要PCA
        else:
            st.session_state.df_pca=st.session_state.encoded_df
    st.success('Completed!')
    st.download_button(label="下载PCA后的数据",data=st.session_state.df_pca.to_csv(index=False).encode('utf-8'),file_name="pca.csv",mime='text/csv')

    if 'balance_data' not in st.session_state:
        st.session_state.balance_data=True#默认需要平衡数据
    if 'start_training' not in st.session_state:
        st.session_state.start_training=False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained=False
    if 'is_binary' not in st.session_state:#判断预测结果是不是二分类
        st.session_state.is_binary=st.session_state.df_pca[st.session_state.selected_Y].nunique()==2

    #8、分析测试集占比
    if 'test_percentage' not in st.session_state:
        with st.spinner('AI分析测试集占比中...'):
            st.session_state.test_percentage=int(decide_test_ratio(st.session_state.df_pca.shape,MODEL,API_KEY)*100)
    splitting_column,balance_column=st.columns(2)
    with splitting_column:
        st.subheader('数据拆分')
        st.caption('AI推荐的模型测试比例')
        st.slider(
            label='测试集比例',
            min_value=1,max_value=25,
            key='test_percentage',#滑动滑块自动赋值给st.session_state.test_percentage
            disabled=st.session_state.start_training
        )
    with balance_column:
        st.metric(label='测试数据',value=f"{st.session_state.test_percentage}%",delta=None)
        st.toggle(label='Class Balancing',key="balance_data",disabled=st.session_state.start_training)
        st.caption('处理不平衡数据集和提高机器学习模型性能的策略')
        st.caption('AI将选择最合适的方法来平衡数据')
    st.button('开始训练模型',on_click=start_train_model,type='primary',disabled=st.session_state.start_training)

    #9、模型训练
    if st.session_state.start_training:
        with st.container():
            st.header('Modeling')
            X,Y=select_Y(st.session_state.df_pca,st.session_state.selected_Y)

            #数据平衡
            if st.session_state.balance_data and 'balance_method' not in st.session_state:
                with st.spinner("AI正在决定数据的平衡策略..."):
                    shape_info_balance,description_info_balance,balance_info_balance=get_balance_info(st.session_state.df_pca,st.session_state.selected_Y)
                    st.session_state.balance_method=int(decide_balance(shape_info_balance,description_info_balance,balance_info_balance,MODEL,API_KEY)['method'])
                    print(f"平衡策略:{st.session_state.balance_method}")
                    X_train_res,Y_train_res=check_and_balance(X,Y,method=st.session_state.balance_method)
            else:
                X_train_res,Y_train_res=X,Y
                if 'balance_method' not in st.session_state:
                    st.session_state.balance_method=4

            #拆分test,train
            if not st.session_state.get('data_splitted',False):
                st.session_state.X_train,st.session_state.X_test,st.session_state.Y_train,st.session_state.Y_test=split_data(X_train_res,Y_train_res,st.session_state.test_percentage/100,perform_pca=st.session_state.to_perform_pca)
                st.session_state.data_splitted=True

            #决定训练模型
            if 'decided_model' not in st.session_state:
                st.session_state.decided_model=False
            if 'all_set' not in st.session_state:
                st.session_state.all_set=False
            if not st.session_state.decided_model:
                with st.spinner("根据数据决定模型..."):
                    shape_info,head_info,nunique_info,description_info=get_data_overview(st.session_state.df_pca)
                    model_dict=decide_model(shape_info,head_info,nunique_info,description_info,MODEL,API_KEY)#{model1:1,model2:2,model3:3}
                    model_list=get_selected_models(model_dict)#[1,2,2]
                    print(f"model_list:{model_list}")
                    if 'model_list' not in st.session_state:
                        st.session_state.model_list=model_list
                    st.session_state.decided_model=True

            if st.session_state.decided_model:
                display_results(st.session_state.X_train,st.session_state.X_test,st.session_state.Y_train,st.session_state.Y_test)
                st.session_state.all_set=True

            if st.session_state.all_set:
                download_col1, download_col2, download_col3 = st.columns(3)
                with download_col1:
                    st.download_button(label="Download Model", data=st.session_state.downloadable_model1,
                                       file_name=f"{st.session_state.model1_name}.joblib",
                                       mime="application/octet-stream")
                with download_col2:
                    st.download_button(label="Download Model", data=st.session_state.downloadable_model2,
                                       file_name=f"{st.session_state.model2_name}.joblib",
                                       mime="application/octet-stream")
                with download_col3:
                    st.download_button(label="Download Model", data=st.session_state.downloadable_model3,
                                       file_name=f"{st.session_state.model3_name}.joblib",
                                       mime="application/octet-stream")
    st.divider()
    if "all_set" in st.session_state and st.session_state.all_set:
        if 'has_been_set' not in st.session_state:
            st.session_state.has_been_set=True

def display_results(X_train,X_test,Y_train,Y_test):
    st.success("模型已选择完毕")

    data_col1,data_col2,data_col3,balance_col4=st.columns(4)
    with data_col1:
        st.metric(label="Total Data",value=len(X_train)+len(X_test),delta=None)
    with data_col2:
        st.metric(label="Train Data",value=len(X_train),delta=None)
    with data_col3:
        st.metric(label="Test Data",value=len(X_test),delta=None)
    with balance_col4:
        st.metric(label="Balance Strategy",value=get_balance_method_name(st.session_state.balance_method),delta=None)

    model_col1,model_col2,model_col3=st.columns(3)
    with model_col1:
        if 'model1_name' not in st.session_state:
            st.session_state.model1_name=get_model_name(st.session_state.model_list[0])
        st.subheader(st.session_state.model1_name)
        with st.spinner("模型训练中..."):
            if 'model1' not in st.session_state:
                st.session_state.model1=train_selected_model(X_train,Y_train,st.session_state.model_list[0])
                st.session_state.downloadable_model1=save_model(st.session_state.model1)
        st.write(f"准确率:\n:green[**{st.session_state.model1.score(X_test,Y_test)}**]")
        st.pyplot(confusion_metrix(st.session_state.model1_name,st.session_state.model1,X_test,Y_test))
        st.write(f"F1 Score:\n:green[**{calculate_f1_score(st.session_state.model1, X_test, Y_test, st.session_state.is_binary)}**]")
        #不是SVC且是二分类时计算FPR/TPR.用于画 ROC 曲线 / 算 AUC
        if st.session_state.model_list[0]!=2 and st.session_state.is_binary:
            if 'fpr1' not in st.session_state:
                fpr1,tpr1=fpr_and_tpr(st.session_state.model1,X_test,Y_test)
                st.session_state.fpr1=fpr1
                st.session_state.tpr1=tpr1
            st.pyplot(roc(st.session_state.model1_name,st.session_state.fpr1,st.session_state.tpr1))
            st.write(f"AUC:\n:green[**{auc(st.session_state.fpr1,st.session_state.tpr1)}**]")

    with model_col2:
        if "model2_name" not in st.session_state:
            st.session_state.model2_name = get_model_name(st.session_state.model_list[1])
        st.subheader(st.session_state.model2_name)
        with st.spinner("模型训练中..."):
            if 'model2' not in st.session_state:
                st.session_state.model2 = train_selected_model(X_train, Y_train, st.session_state.model_list[1])
                st.session_state.downloadable_model2 = save_model(st.session_state.model2)
        # Model metrics
        st.write(f"准确率: \n:green[**{st.session_state.model2.score(X_test, Y_test)}**]")
        st.pyplot(confusion_metrix(st.session_state.model2_name, st.session_state.model2, X_test, Y_test))
        st.write(f"F1 Score:\n:green[**{calculate_f1_score(st.session_state.model2, X_test, Y_test, st.session_state.is_binary)}**]")
        if st.session_state.model_list[1] != 2 and st.session_state['is_binary']:
            if 'fpr2' not in st.session_state:
                fpr2, tpr2 = fpr_and_tpr(st.session_state.model2, X_test, Y_test)
                st.session_state.fpr2 = fpr2
                st.session_state.tpr2 = tpr2
            st.pyplot(roc(st.session_state.model2_name, st.session_state.fpr2, st.session_state.tpr2))
            st.write(f"AUC:\n:green[**{auc(st.session_state.fpr2, st.session_state.tpr2)}**]")

    with model_col3:
        if "model3_name" not in st.session_state:
            st.session_state.model3_name = get_model_name(st.session_state.model_list[2])
        st.subheader(st.session_state.model3_name)
        with st.spinner("模型训练中..."):
            if 'model3' not in st.session_state:
                st.session_state.model3 = train_selected_model(X_train, Y_train, st.session_state.model_list[2])
                st.session_state.downloadable_model3 = save_model(st.session_state.model3)
        # Model metrics
        st.write(f"准确率:\n:green[**{st.session_state.model3.score(X_test, Y_test)}**]")
        st.pyplot(confusion_metrix(st.session_state.model3_name, st.session_state.model3, X_test, Y_test))
        st.write(f"F1 Score:\n:green[**{calculate_f1_score(st.session_state.model3, X_test, Y_test, st.session_state.is_binary)}**]")
        if st.session_state.model_list[2] != 2 and st.session_state['is_binary']:
            if 'fpr3' not in st.session_state:
                fpr3, tpr3 = fpr_and_tpr(st.session_state.model3, X_test, Y_test)
                st.session_state.fpr3 = fpr3
                st.session_state.tpr3 = tpr3
            st.pyplot(roc(st.session_state.model3_name, st.session_state.fpr3, st.session_state.tpr3))
            st.write(f"AUC:\n:green[**{auc(st.session_state.fpr3, st.session_state.tpr3)}**]")