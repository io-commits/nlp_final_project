import streamlit as st
import pandas as pd

st.markdown(
    """
<style>
.css-1v3fvcr {
    background-color: ghostwhite !important;
    height: 100vh;
    width: 100vw;
    }

</style>
""",
    unsafe_allow_html=True,
)

nlp = '<p style="text-align:center; color:black; font-size: 20px;">NLP final project</p>'
st.markdown(nlp, unsafe_allow_html=True)

title = '<p style="font-family:Courier; text-align:center; color:Blue; font-size: 40px;">Ukraine-Russia conflict tweet text generator</p>'
st.markdown(title, unsafe_allow_html=True)

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

data_load_state = st.text('Loading data...')
data = load_data(10000)
data_load_state.text("Done! (using st.cache)")

text_placeholder = st.empty()
header = '<p style="color:darkred; font-size: 15px;">*Fill all the fields before run:</p>'
text_placeholder.markdown(header, unsafe_allow_html=True)

radio_placeholder = st.empty()
text_input_placeholder = st.empty();

option = radio_placeholder.radio(
     "Please choose generation or autocomplete",
     ('generation', 'autocomplete'))
if option == 'autocomplete':
    text_input_placeholder.text_input('You choose on auto_complete model, Please insert sequence to start with')

# if g_a_form .form_submit_button('Select'):
placeholder = st.empty()
form = placeholder.form("my_form")
form.selectbox('Please select model', ['model1', 'model2', 'model3', 'model4', 'model5'])
form.text('Please inferred amount of characters')
number = '<input type="number" min="0" max="100" placeholder="" ' \
         'style="color:green; width: 42vw; height: 7vh;"></input>'
form.markdown(number, unsafe_allow_html=True)
form.select_slider('Please select how many words to generate in the tweet',
                   ['top10k', 'top50k', 'top100k', 'top1m', 'top1ksampled', 'Random :)'])

if form.form_submit_button('Generate Some Text'):
    placeholder.empty()
    text_placeholder.empty()
    radio_placeholder.empty()
    text_input_placeholder.empty()
    text = "'tweet data: bla bla...................................'"
    st.write(text)

    st.title('evaluation:')
    col1, col2 = st.columns(2)
    col1.metric("One params??", "0.5", "-0.6")
    col2.metric("Two Params?? ", "90%", "80%")








