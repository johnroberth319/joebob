import streamlit as st

pages = {
    "Personal": [
        st.Page("Home.py", title="Home Page")
    ],
    "GCU - CST435": [
        st.Page("Assignment2.py", title="Assignment 2"),
<<<<<<< HEAD
        st.Page("Assignment2/A2.py", title="Assignment 2 - Main")
=======
        st.Page("Assignment2\MAIN.py", title="Assignment 2 - Main")
>>>>>>> parent of 4c9b96e (Update streamlit_app.py)
    ],
}

def main():
    pg = st.navigation(pages)
    pg.run()

if __name__ == "__main__":
    main()