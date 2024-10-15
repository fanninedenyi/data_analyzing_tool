import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Title of the app
st.title('Multiple Variable Analysis Between Two Groups')

# Step 1: Upload Excel File
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    # Step 2: Read the Excel file
    df = pd.read_excel(uploaded_file)

    # Display the dataframe
    st.write("Data Preview:")
    st.dataframe(df)

    # Check if the data contains the expected columns
    if 'Group' in df.columns:
        # Select the numeric variables for testing
        numeric_columns = df.select_dtypes(include='number').columns.tolist()

        if len(numeric_columns) > 0:
            st.write(f"Numeric variables detected: {', '.join(numeric_columns)}")

            # Loop over all numeric variables and perform analysis
            results = []
            for col in numeric_columns:
                st.write(f"### Analysis for Variable: {col}")

                # Step 3: Visualize the data for each variable with a boxplot
                fig, ax = plt.subplots()
                df.boxplot(column=col, by='Group', ax=ax)
                plt.title(f'Boxplot of {col} by Group')
                plt.suptitle('')  # Suppress the default title
                st.pyplot(fig)

                # Step 4: Perform t-test for each variable
                group_a = df[df['Group'] == 'Group A'][col]
                group_b = df[df['Group'] == 'Group B'][col]

                t_stat, p_val = ttest_ind(group_a, group_b, nan_policy='omit')

                st.write(f"T-statistic for {col}: {t_stat:.4f}")
                st.write(f"P-value for {col}: {p_val:.4f}")

                # Step 5: Conclusion for each variable
                alpha = 0.05  # Significance level
                if p_val < alpha:
                    st.write(f"Conclusion: The variable '{col}' differs significantly between the groups (Reject the null hypothesis).")
                else:
                    st.write(f"Conclusion: The variable '{col}' does not differ significantly between the groups (Fail to reject the null hypothesis).")

                # Store the result
                results.append({
                    'Variable': col,
                    'T-statistic': t_stat,
                    'P-value': p_val,
                    'Significant Difference': 'Yes' if p_val < alpha else 'No'
                })

            # Step 6: Display overall results summary in a table
            st.write("### Summary of Results")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)

        else:
            st.error("No numeric variables found in the dataset.")
    else:
        st.error("The Excel file must contain a 'Group' column.")
