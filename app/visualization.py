import streamlit as st

from src.plot import count_Y, violin_plot, distribution_histogram, box_plot, density_plot, strip_plot, \
    distribution_boxplot, multi_plot_heatmap, multi_plot_line, multi_plot_scatter, scatter_3d, word_cloud_plot, list_all

"""
plot_type1
plot_type2
"""

def display_word_cloud(text):
    _,word_cloud_col,_=st.columns([1,3,1])
    with word_cloud_col:
        word_fig=word_cloud_plot(text)
        if word_fig==-1:
            st.error('数据不支持')
        else:
            st.pyplot(word_fig)

def data_visualization(DF_uploaded):
    st.divider()
    st.subheader('数据可视化')
    attributes=DF_uploaded.columns
    st.write(DF_uploaded.dtypes)

    single_tab,multiple_tab,advanced_tab=st.tabs(['单一属性可视化','多属性可视化','高级可视化'])

    with single_tab:
        _,col_mid,_=st.columns([1,5,1])
        #给下面的图表空着
        with col_mid:
            plot_area=st.empty()
        col1,col2=st.columns(2)
        with col1:
            att=st.selectbox(label='选择一个属性用于可视化',options=attributes,index=len(attributes)-1)
            st.write(f"选择的属性: :green[{att}]")
        with col2:
            plot_types=['环形图','小提琴图','分布直方图','箱线图','密度图','条形图','分布箱线图']
            plot_type=st.selectbox(
                label='选择一个图标类型',
                options=plot_types,
                index=0,
                key='plot_type1'
            )
            st.write(f'选择的图表类型: :green[{plot_type}]')
        if plot_type=='环形图':
            fig = count_Y(DF_uploaded, att)
            plot_area.plotly_chart(fig)
        elif plot_type=='小提琴图':
            fig = violin_plot(DF_uploaded, [att])
            plot_area.plotly_chart(fig)
        elif plot_type=='分布直方图':
            fig = distribution_histogram(DF_uploaded, att)
            plot_area.pyplot(fig)
        elif plot_type=='箱线图':
            fig = box_plot(DF_uploaded, [att])
            plot_area.plotly_chart(fig)
        elif plot_type=='密度图':
            fig = density_plot(DF_uploaded, att)
            plot_area.plotly_chart(fig)
        elif plot_type=='条形图':
            fig = strip_plot(DF_uploaded, [att])
            plot_area.plotly_chart(fig)
        elif plot_type=='分布箱线图':
            fig = distribution_boxplot(DF_uploaded, att)
            if fig==-1:
                plot_area.error('属性不是数值型')
            else:
                plot_area.pyplot(fig)

    with multiple_tab:
        col1, col2 = st.columns([6, 4])
        with col1:
            options=st.multiselect(
                label="选择多个属性可视化",
                options=attributes,
                default=[]
            )
        with col2:
            plot_types=['小提琴图','箱线图','条形图','热力图','线性图','散点图']
            plot_type=st.selectbox(
                key='plot_type2',
                label="选择一个图表类型",
                options=plot_types,
                index=0,
            )
        _,col_mid,_=st.columns([1,5,1])
        with col_mid:
            plot_area=st.empty()
        if options:
            if plot_type=='小提琴图':
                fig=violin_plot(DF_uploaded, options)
                if fig==-1:
                    plot_area.error("存在非数值属性")
                else:
                    plot_area.plotly_chart(fig)
            elif plot_type=='箱线图':
                fig=box_plot(DF_uploaded, options)
                if fig==-1:
                    plot_area.error('存在非数值属性')
                else:
                    plot_area.plotly_chart(fig)
            elif plot_type=='条形图':
                fig=strip_plot(DF_uploaded, options)
                if fig==-1:
                    plot_area.error('存在非数值属性')
                else:
                    plot_area.plotly_chart(fig)
            elif plot_type=='热力图':
                fig=multi_plot_heatmap(DF_uploaded, options)
                if fig==-1:
                    plot_area.error('存在非数值属性')
                else:
                    plot_area.pyplot(fig)
            elif plot_type=='线性图':
                fig=multi_plot_line(DF_uploaded, options)
                if fig==-1:
                    plot_area.error('存在非数值属性')
                elif fig==-2:
                    plot_area.error('Line plot要求至少两个属性')
                else:
                    plot_area.pyplot(fig)
            elif plot_type=='散点图':
                fig=multi_plot_scatter(DF_uploaded, options)
                if fig==-1:
                    plot_area.error('散点图要两个属性')
                else:
                    plot_area.pyplot(fig)

    with advanced_tab:
        st.subheader('3D散点图')
        column_1, column_2, column_3 = st.columns(3)
        with column_1:
            x=st.selectbox(
                key='x',
                label='选择x属性',
                options=attributes,
                index=0
            )
        with column_2:
            y=st.selectbox(
                key='y',
                label='选择y属性',
                options=attributes,
                index=1 if len(attributes)>1 else 0
            )
        with column_3:
            z=st.selectbox(
                key='z',
                label='选择z属性',
                options=attributes,
                index=2 if len(attributes)>2 else 0
            )
        if st.button('生成3D图表'):
            _,fig_3d_col,_=st.columns([1, 3, 1])
            with fig_3d_col:
                fig_3d_1=scatter_3d(DF_uploaded,x,y,z)
                if fig_3d_1==-1:
                    st.error('数据不支持')
                else:
                    st.plotly_chart(fig_3d_1)
        st.divider()

        st.subheader('词云')
        upload_txt_checkbox=st.checkbox('上传一个新的文本文件')
        if upload_txt_checkbox:
            upload_txt=st.file_uploader('选择一个文本文件',accept_multiple_files=False,type='txt')
            if upload_txt:
                text=upload_txt.read().decode('utf-8')
                display_word_cloud(text)
        else:
            text_attr=st.selectbox(
                label='选择一个文本属性',
                options=attributes,
                index=0
            )
            if st.button('生成词云'):
                text=DF_uploaded[text_attr].astype(str).str.cat(sep=' ')
                display_word_cloud(text)
        st.divider()

    st.subheader('数据展示')
    if 'data_origin' not in st.session_state:
        st.session_state.data_origin=DF_uploaded
    st.dataframe(st.session_state.data_origin.describe(),width=1200)
    if 'overall_plot' not in st.session_state:
        st.session_state.overall_plot=list_all(st.session_state.data_origin)
    st.pyplot(st.session_state.overall_plot)
    st.divider()