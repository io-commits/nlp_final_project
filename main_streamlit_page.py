import streamlit as st


st.title('Ukraine-Russia conflict tweet text generator')
st.image('res/Volodymyr_Zelensky_Official_portrait.jpg', width=200)
st.image('res/Vladimir_Putin_17-11-2021_(cropped).jpg', width=200)

st.header('Inference')
st.subheader('Data the model has trained with')
st.multiselect('', ['top10k', 'top50k', 'top100k', 'top1m', 'top1ksampled'])
st.number_input('Inferred amount of characters', 0, 100)
st.text_input('Please insert sequence to start with')

