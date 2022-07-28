import subprocess
from pathlib import Path

import streamlit as st
from model.gpt_neo import GptNeo
from model.baseline import Baseline

from commander import Commander
from evaluate import evaluation_metrics
com = Commander()

sample_control_states = ['sample_text_area', 'amount_of_words_passed', 'seed']
autocomplete_control_states = ['auto_input_text', 'auto_input_number']


st.session_state.sample_text_area = False
st.session_state.seed_text_area = False


st.session_state.auto_input_text = False
st.session_state.auto_input_number = 0


sb = st.selectbox('Please select model', ['gpt_neo_sampled30m_no_hashtags',
                                          'gpt_neo_sampled10m_no_hashtags',
                          'gpt_neo_sampled1m',
                          'gpt_neo_sample2m',
                          'baseline1',
                          'baseline2'])

model = com.get(sb)
sample, autocomplete = st.tabs(['Sample', 'Completion'])
if sample:
    sample.text('You choose on sample model, please hit the Sample button below to select sentence')
    ret = com.sample()
    while len(ret) <=10:
        ret = com.sample()
    sample.text_area('Sampled', ret)
    seed = ret[:len(ret)//2]
    sample.text_area('Seed is', seed)
    if 'gpt' in sb:
        sampled_text = model.generate(seed, max_length=len(ret)//2)
        sample.write(sampled_text)
        lst = evaluation_metrics([ret], [f'{seed}{sampled_text}'])
        colrough, colter = sample.columns(2)
        colrough.text('Recall focused - the higher the better')
        colter.text('Edit focused - the lower the better')
        col1, col2 = sample.columns(2)
        col1.metric(label=f'rough', value=lst[1])
        col2.metric(label=f'ter', value=lst[2])

    if st.button('refresh'):
        pass

if autocomplete:
    autocomplete.text('You choose on auto_complete model')
    input_text = autocomplete.text_input(' Please insert sequence to start with')
    st.session_state.auto_input_number = autocomplete.number_input('Character to predict', 1, 40)
    if autocomplete.button('Predict Autocomplete'):
        if 'gpt' in sb:
            text = model.generate(input_text, max_length=st.session_state.auto_input_number)
            autocomplete.write(text)

