import streamlit as st

# Title
st.title("Simple Streamlit Test")

# Text input field
user_input = st.text_input("Enter some text:", "Type here...")

# Button to display the entered text
if st.button("Display Text"):
    st.write(f"You entered: {user_input}")