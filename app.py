import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Title of the app
st.title('Group-wise Variable Analysis')

# Step 1: Upload Excel File
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    # Step 2: Read the Excel file
    df = pd.read_excel(uploaded_file)

    # Display the dataframe
    st.write("Data Preview:")
    st.dataframe(df)

    # Check if the data contains the expected columns
    if 'Group' in df.columns and 'Variable' in df.columns:
        
        # Step 3: Visualize the data with a boxplot
        st.write("Visualizing the Data:")
        fig, ax = plt.subplots()
        df.boxplot(column='Variable', by='Group', ax=ax)
        plt.title('Boxplot of Variable by Group')
        plt.suptitle('')  # Suppress the default title
        st.pyplot(fig)

        # Step 4: Perform t-test
        st.write("Performing t-test:")
        group_a = df[df['Group'] == 'Group A']['Variable']
        group_b = df[df['Group'] == 'Group B']['Variable']

        # Perform independent t-test
        t_stat, p_val = ttest_ind(group_a, group_b)

        st.write(f"T-statistic: {t_stat:.4f}")
        st.write(f"P-value: {p_val:.4f}")

        # Step 5: Conclusion
        alpha = 0.05  # Significance level
        if p_val < alpha:
            st.write("Conclusion: The variable depends on the group (Reject the null hypothesis).")
        else:
            st.write("Conclusion: The variable does not depend on the group (Fail to reject the null hypothesis).")

    else:
        st.error("The Excel file must contain columns 'Group' and 'Variable'.")
