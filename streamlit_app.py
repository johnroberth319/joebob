import streamlit as st

pages = {
    "Personal": [
        st.Page("Home.py", title="Home Page")
    ],
    "GCU": [
        "CST435 - Deep Learning": [
            st.Page("CST435.py", title="CST435 - Deep Learning")
        ]
    ],
}

def main():
    pg = st.navigation(pages)
    pg.run()

if __name__ == "__main__":
    main()