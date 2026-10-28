import json
import os
import re

import yaml
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
import streamlit as st

config_path=os.path.join(os.path.dirname(__file__),'config','config.yaml')
with open(config_path,'r',encoding='utf-8') as f:
    config_data=yaml.safe_load(f)

@st.cache_resource
def get_llm(model,api_key):
    if model.startswith('ollama'):
        model_name = model.replace('ollama:', '')
        return ChatOllama(model=model_name, temperature=0)
    elif model.startswith('deepseek'):
        model_name = model.replace('deepseek:', '')
        return ChatOpenAI(model=model_name, base_url=config_data['deepseek_base_url'], api_key=api_key, temperature=0)
    else:
        raise ValueError(f"Unsupported model: {model}")

def decide_target_attribute(attributes:list[str],types_info:str,head_info:str,model:str,api_key:str|None=None):
    """
    找出目标属性
    :param attributes:
    :param types_info:
    :param head_info:
    :param model:
    :param api_key:
    :return: 目标属性名称
    """
    try:
        llm=get_llm(model,api_key)
        #组合提示词
        template=config_data['decide_target_attribute_template']
        prompt_template=PromptTemplate.from_template(template)
        summary_prompt=prompt_template.format(attributes=attributes,types_info=types_info,head_info=head_info)

        print(f"调用大模型(找寻目标特征值)...")
        print(f"llm:{llm}")
        llm_answer:AIMessage=llm.invoke([HumanMessage(content=summary_prompt)])
        print("CONTENT =", repr(llm_answer.content))
        print("ADDITIONAL =", llm_answer.additional_kwargs)
        print("METADATA =", llm_answer.response_metadata)
        #llm_answer.content={'target':'属性名' or -1}
        #返回的json被markdown包裹了一层
        if '```json' in llm_answer.content:
            #从里面提取出主要内容json
            match=re.search(r'```json\n(.*?)```',llm_answer.content,re.DOTALL)
            if match:
                json_str=match.group(1)
        else:
            json_str=llm_answer.content
        return json.loads(json_str)["target"]
    except Exception as e:
        import traceback
        traceback.print_exc()  # 终端看完整堆栈
        st.error(f"{type(e).__name__}: {e}")
        st.error(f"Please check your api_key or network connection：{str(e)}")
        st.stop()

def decide_fill_null(attributes:list[str],types_info:str,description_info:str,model:str,api_key:str|None=None)->dict[str,int]:
    """
    根据列名，数据类型，数据总结(count,mean,std...)返回每一列应该使用哪种填充方式
    {
        "name":4,新类别
        "age":1,均值填充
        "time":5，插值填充
    }
    :param attributes:
    :param types_info:
    :param description_info:
    :param model:
    :param api_key:
    :return:
    """
    try:
        llm=get_llm(model, api_key)
        template=config_data['null_attribute_template']
        prompt_template=PromptTemplate.from_template(template)
        summary_prompt=prompt_template.format(attributes=attributes,types_info=types_info,description_info=description_info)

        print("开始请求大模型(列的填充方式)...")
        print(f"llm:{llm}")
        llm_answer:AIMessage=llm.invoke([HumanMessage(content=summary_prompt)])
        print(f"llm_answer.content:{llm_answer.content}")
        if '```json' in llm_answer.content:
            match=re.search(r'```json\n(.*?)```',llm_answer.content,re.DOTALL)
            if match:
                json_str=match.group(1)
        else:
            json_str=llm_answer.content
        return json.loads(json_str)
    except Exception as e:
        import traceback
        traceback.print_exc()  # 完整堆栈，打到终端
        st.error(f"{type(e).__name__}: {e}")
        st.error(f"Please check your API key or network connection.:str{e}")
        st.stop()

def decide_encode_type(attributes_unique_info:dict[str,int],model:str,api_key:str|None)->dict[str,int]:
    """

    Args:
        attributes_unique_info:
        model:
        api_key:

    Returns:
        1:数值映射
        2:one-hot
        3:舍弃
        {
            name:2,
            gender:1,
            comment:3
        }
    """
    try:
        llm=get_llm(model, api_key)

        template=config_data['numeric_attributes_template']
        prompt_template=PromptTemplate.from_template(template)
        summary_prompt=prompt_template.format(attributes_unique_info=attributes_unique_info)

        llm_answer=llm.invoke([HumanMessage(content=summary_prompt)])
        if '```json' in llm_answer.content:
            match=re.search(r'```json\n(.*?)```',llm_answer.content,re.DOTALL)
            if match:
                json_str=match.group(1)
        else:
            json_str=llm_answer.content
        return json.loads(json_str)
    except Exception as e:
        st.error("Please check your API key or network connection.")
        st.stop()

def decide_test_ratio(shape_info,model:str,api_key:str|None):
    try:
        llm=get_llm(model, api_key)

        template=config_data['decide_test_ratio_template']
        prompt_template=PromptTemplate.from_template(template)
        summary_prompt=prompt_template.format(shape_info=shape_info)

        llm_answer=llm.invoke([HumanMessage(content=summary_prompt)])
        if '```json' in llm_answer.content:
            match=re.search(r'```json\n(.*?)```',llm_answer.content,re.DOTALL)
            if match:
                json_str=match.group(1)
        else:
            json_str=llm_answer.content
        return json.loads(json_str)['test_ratio']
    except Exception as e:
        st.error("Please check your API key or network connection.")
        st.stop()

def decide_balance(shape_info:tuple[int,int],description_info:str,balance_info:dict[int,int],model:str,api_key:str|None):
    """
    根据数据集的特征确定平衡数据集的适当方法。
    :param shape_info:
    :param description_info:
    :param balance_info:
    :param model:
    :param api_key:
    :return:
    """
    try:
        llm=get_llm(model, api_key)
        template=config_data['decide_balance_template']
        prompt_template=PromptTemplate.from_template(template)
        summary_prompt=prompt_template.format(shape_info=shape_info,description_info=description_info,balance_info=balance_info)

        llm_answer=llm.invoke([HumanMessage(content=summary_prompt)])
        if '```json' in llm_answer.content:
            match=re.search(r'```json\n(.*?)```',llm_answer.content,re.DOTALL)
            if match:
                json_str=match.group(1)
        else:
            json_str=llm_answer.content
        return json.loads(json_str)
    except Exception as e:
        st.error("Please check your API key or network connection.")
        st.stop()

def decide_model(shape_info,head_info,nunique_info,description_info,model:str,api_key:str|None):
    """
    根据数据集特征确定最合适的机器学习模型。
    :param shape_info:
    :param head_info:
    :param nunique_info:
    :param description_info:
    :param model:
    :param api_key:
    :return:
    """
    try:
        llm=get_llm(model, api_key)
        template=config_data['decide_model_template']
        prompt_template=PromptTemplate.from_template(template)
        summary_prompt=prompt_template.format(shape_info=shape_info,head_info=head_info,nunique_info=nunique_info,description_info=description_info)
        llm_answer=llm.invoke([HumanMessage(content=summary_prompt)])
        if '```json' in llm_answer.content:
            match=re.search(r'```json\n(.*?)```',llm_answer.content,re.DOTALL)
            if match:
                json_str=match.group(1)
        else:
            json_str=llm_answer.content
        return json.loads(json_str)
    except Exception as e:
        st.error("Please check your API key or network connection.")
        st.stop()