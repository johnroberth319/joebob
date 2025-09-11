import streamlit as st

def main():
    st.sidebar.title("Navigation")

    menu = st.sidebar.selectbox("Choose a category", ["Personal", "GCU"])

    if menu == "Personal":
        page = st.sidebar.radio("Select a page", ["Home"])
        if page == "Home":
            st.write("This is the Home Page.")

    elif menu == "GCU":
        course = st.sidebar.selectbox("Choose a course", ["CST435"])
        if course == "CST435":
            page = st.sidebar.radio("Select a CST435 page", ["Overview", "Assignments", "Labs"])
            
            if page == "Overview":
                st.write("CST435 Overview content")
            elif page == "Assignments":
                st.write("CST435 Assignments content")
            elif page == "Labs":
                st.write("CST435 Labs content")

if __name__ == "__main__":
    main()
