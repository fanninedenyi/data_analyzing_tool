import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title of the app
st.title('Excel Data Visualizer')

# Step 1: Upload Excel File
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

if uploaded_file:
    # Step 2: Read the Excel file
    df = pd.read_excel(uploaded_file)

    # Display the dataframe
    st.write("Data Preview:")
    st.dataframe(df)

    # Step 3: Select columns to visualize
    st.write("Data Visualization")
    columns = df.columns.tolist()
    
    # Choose columns for visualization
    x_axis = st.selectbox('Choose X-axis', columns)
    y_axis = st.selectbox('Choose Y-axis', columns)

    # Step 4: Plot the data
    if st.button('Generate Plot'):
        plt.figure(figsize=(10, 5))
        plt.scatter(df[x_axis], df[y_axis])
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title(f'{y_axis} vs {x_axis}')
        st.pyplot(plt)
