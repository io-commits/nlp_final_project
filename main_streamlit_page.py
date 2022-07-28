import streamlit as st
from commander import Commander
from evaluate import evaluation_metrics

# initializes commander
com = Commander()

# holding the number of autocomplete characters amount
st.session_state.auto_input_number = 0

st.title('Ukraine - Russia conflict tweet generator')

st.header('Model Selection')

sb = st.selectbox('Please select model', ['gpt_neo_sampled30m_no_hashtags',
                                          'gpt_neo_sampled10m_no_hashtags',
                                          'gpt_neo_sampled1m',
                                          'gpt_neo_sample2m'])

# get current model
model = com.get(sb)

st.header('Task Selection')

# initialize two tabs
sample, autocomplete = st.tabs(['Sample', 'Completion'])

# this section handles the probabilistic sample pick
if sample:
    sample.text('You choose on sample task.\nWe pick randomly a trained sequence and predict a half of it.')
    ret = com.sample()
    while len(ret) <= 200:
        ret = com.sample()
    sample.text_area('Sampled sequence', ret)
    seed = ret[:len(ret) // 2]
    sample.text_area('Seed is half of the input', seed)
    # constructed with baselines models in plans which not currently supported
    # left as is as a template
    if 'gpt' in sb:
        sampled_text = model.generate(seed, max_length=len(ret) // 2)
        sample.write(sampled_text)
        lst = evaluation_metrics([ret], [f'{seed}{sampled_text}'])
        colbleu, colrough, colter = sample.columns(3)
        colbleu.title('Precision focused - the higher the better')
        colrough.title('Recall focused - the higher the better')
        colter.title('Edit focused - the lower the better')
        col0, col1, col2 = sample.columns(3)
        col0.metric(label=f'bleu', value=lst[0])
        col1.metric(label=f'rough', value=lst[1])
        col2.metric(label=f'ter', value=lst[2])

    if st.button('refresh'):
        pass

# this section handles the autocomplete
if autocomplete:
    autocomplete.text('You choose on completion task.')
    autocomplete.text('Please pay attention for hitting ENTER after you insert your decisions.')
    input_text = autocomplete.text_input(' Please insert sequence to start with')
    st.session_state.auto_input_number = \
        autocomplete.number_input('Please insert the maximum number of generated tokens', 1, 100)
    if autocomplete.button('Predict consequence tokens'):
        if 'gpt' in sb:
            text = model.generate(input_text, max_length=st.session_state.auto_input_number)
            autocomplete.write(text)
