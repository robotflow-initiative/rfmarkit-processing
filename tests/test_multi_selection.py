import streamlit as st
st.write("Hello world")

label = "IMU Friendly Name"
options = ["A-1", "A-2", "A-3", "A-4"]

selection = ["" for _ in range(10)]
for i in range(len(selection)):
    selection[i] = st.radio(label, options, index=0, key=i, horizontal=True)

print(selection)
