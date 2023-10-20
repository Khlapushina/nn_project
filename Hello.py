import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Our App! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
  
    **👈 Select an app from the sidebar**

"""
)