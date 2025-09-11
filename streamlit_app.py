import streamlit as st

st.set_page_config(
    page_title="JoeBob - John's Streamlit Portfolio",
    page_icon="🐐",
    layout="wide"
)

st.title("JoeBob")
def page_2():
    st.title("CST435 - Deep Learning")

pg = st.sidebar.selectbox("Select your page", ["Home", "CST435 - Deep Learning"])
if pg == "CST435 - Deep Learning":
    page_2()

pg.run()