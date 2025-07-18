# my_app.py
import streamlit as st
import pandas as pd

# 1. Add a title
st.title("My First Streamlit App 🎈")

# 2. Add some text
st.write("This is a simple app to demonstrate Streamlit's features.")

# 3. Add a slider widget
age = st.slider("Select your age:", 0, 100, 25)
st.write(f"You selected: **{age}**")

# 4. Display a DataFrame
st.subheader("Displaying Data")
df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
})
st.write(df)

# 5. Add a button
if st.button("Click Me!"):
    st.success("You clicked the button! 🎉")
