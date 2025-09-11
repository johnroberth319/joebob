import streamlit as st

def main():
    st.sidebar.title("Navigation")

    menu = st.sidebar.selectbox("Choose a category", ["Personal", "GCU"])

    if menu == "Personal":
        page = st.sidebar.radio("Select a page", ["Home"])
        if page == "Home":
            st.Page("Home.py", title="Home Page", icon="🏠")

    elif menu == "GCU":
        course = st.sidebar.selectbox("Choose a course", ["CST435"])
        if course == "CST435":
            page = st.sidebar.radio("Select a CST435 page", ["Overview", "Assignments", "Labs"])

if __name__ == "__main__":
    main()
