import os
import random
import time
import streamlit as st
import requests
import yaml

config_path=os.path.join(os.path.dirname(__file__),'config','config.yaml')
with open(config_path,'r',encoding='utf-8') as f:
    config_data=yaml.safe_load(f)    #Yaml转python对象

def load_lottie():
    r1,r2=requests.get(config_data['lottie_url1']),requests.get(config_data['lottie_url2'])
    if r1.status_code != 200 or r2.status_code != 200:
        return None
    return r1.json(),r2.json()

def stream_data(line):
    for word in line.split():
        yield word+" "
        time.sleep(random.uniform(0.02,0.05))

def welcome_message():
    return config_data['welcome_template']

def introduction_message():
    return config_data['introduction_template1'],config_data['introduction_template2']
