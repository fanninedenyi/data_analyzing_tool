import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import ttest_ind

# Title of the app
st.title('Variable Analysis Between Two Groups')

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

            num_vars = len(numeric_columns)
            
            if num_vars == 1:
                st.write(f"### Analysis for Variable: {numeric_columns[0]}")

                # Step 3: Visualize the data with a boxplot (1 variable)
                fig, ax = plt.subplots()
                df.boxplot(column=numeric_columns[0], by='Group', ax=ax)
                plt.title(f'Boxplot of {numeric_columns[0]} by Group')
                plt.suptitle('')  # Suppress the default title
                st.pyplot(fig)

            elif num_vars == 2:
                st.write(f"### 2D Scatter Plot for Variables: {numeric_columns[0]} and {numeric_columns[1]}")

                # Step 3: Visualize with a 2D scatter plot (2 variables)
                fig = px.scatter(df, x=numeric_columns[0], y=numeric_columns[1], color='Group',
                                 title=f'{numeric_columns[0]} vs {numeric_columns[1]}')
                st.plotly_chart(fig)

            elif num_vars == 3:
                st.write(f"### 3D Scatter Plot for Variables: {numeric_columns[0]}, {numeric_columns[1]}, and {numeric_columns[2]}")

                # Step 3: Visualize with a 3D scatter plot (3 variables)
                fig = px.scatter_3d(df, x=numeric_columns[0], y=numeric_columns[1], z=numeric_columns[2], color='Group',
                                    title=f'{numeric_columns[0]} vs {numeric_columns[1]} vs {numeric_columns[2]}')
                st.plotly_chart(fig)

            else:
                st.write("More than 3 numeric variables detected. No plot will be generated.")

            # Perform t-tests for each variable
            st.write("### Performing t-tests for numeric variables:")
            results = []
            for col in numeric_columns:
                group_a = df[df['Group'] == 'Group A'][col]
                group_b = df[df['Group'] == 'Group B'][col]

                t_stat, p_val = ttest_ind(group_a, group_b, nan_policy='omit')

                st.write(f"T-statistic for {col}: {t_stat:.4f}")
                st.write(f"P-value for {col}: {p_val:.4f}")

                # Conclusion for each variable
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

            # Display summary table of t-test results
            st.write("### Summary of t-test Results")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)

        else:
            st.error("No numeric variables found in the dataset.")
    else:
        st.error("The Excel file must contain a 'Group' column.")
