import streamlit as st

st.set_page_config(
    page_title="JoeBob - John's Streamlit Portfolio",
    page_icon="🐐",
    layout="wide"
)

pages = {
    "Personal": [
        st.Page("streamlit_app.py", title="Home Page")
    ],
    "GCU": [
        st.Page("CST435.py", title="CST435 - Deep Learning")
    ],
}

def main():
    st.title("JoeBob")

    pg = st.navigation(pages)
    pg.run()

