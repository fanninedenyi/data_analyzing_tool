import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import ttest_ind

# Title of the app
st.title('Researcher Tool: Guided Variable Analysis')

# Step 1: Upload Excel File
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    # Step 2: Read the Excel file
    df = pd.read_excel(uploaded_file)

    # Display the dataframe
    st.write("Data Preview:")
    st.dataframe(df)

    # Step 3: Feedback on the structure of the dataset
    st.write("### Dataset Information")
    num_columns = df.shape[1]
    st.write(f"Total number of columns: {num_columns}")

    # Identify categorical and numerical columns
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

    st.write(f"Categorical columns: {', '.join(categorical_columns)}")
    st.write(f"Numeric columns: {', '.join(numeric_columns)}")

    # Step 4: Let the user choose the grouping variable
    st.write("### Select a Grouping Variable")
    group_col = st.selectbox("Choose a column to use as the grouping variable", categorical_columns)

    # Step 5: Let the user select variables to include in the study
    st.write("### Select Variables to Include in the Study")
    selected_numeric_cols = []

    # Checkbox for each numeric column to allow the user to include/exclude variables
    for col in numeric_columns:
        include_var = st.checkbox(f"Include {col}", value=False)
        if include_var:
            selected_numeric_cols.append(col)

    # Dynamic message based on the number of selected variables
    num_vars = len(selected_numeric_cols)
    
    if num_vars == 0:
        st.write("Please select at least one numeric variable for analysis.")
    elif num_vars == 1:
        st.write(f"You have selected {num_vars} variable, so we'll generate a boxplot for comparison.")
    elif num_vars == 2:
        st.write(f"You have selected {num_vars} variables, so we'll generate a 2D scatter plot for visualization.")
    elif num_vars == 3:
        st.write(f"You have selected {num_vars} variables, so we'll generate a 3D scatter plot for visualization.")
    else:
        st.write(f"You have selected {num_vars} variables, so dimensionality reduction will be required.")

    # Only proceed if the user selects at least one numeric variable
    if num_vars > 0:
        # Step 6: Data Visualization and Analysis
        if num_vars == 1:
            st.write(f"### Analysis for Variable: {selected_numeric_cols[0]}")

            # Visualize the data with a boxplot (1 variable)
            fig, ax = plt.subplots()
            df.boxplot(column=selected_numeric_cols[0], by=group_col, ax=ax)
            plt.title(f'Boxplot of {selected_numeric_cols[0]} by {group_col}')
            plt.suptitle('')  # Suppress the default title
            st.pyplot(fig)

        elif num_vars == 2:
            st.write(f"### 2D Scatter Plot for Variables: {selected_numeric_cols[0]} and {selected_numeric_cols[1]}")

            # Visualize with a 2D scatter plot (2 variables)
            fig = px.scatter(df, x=selected_numeric_cols[0], y=selected_numeric_cols[1], color=group_col,
                             title=f'{selected_numeric_cols[0]} vs {selected_numeric_cols[1]} by {group_col}')
            st.plotly_chart(fig)

        elif num_vars == 3:
            st.write(f"### 3D Scatter Plot for Variables: {selected_numeric_cols[0]}, {selected_numeric_cols[1]}, and {selected_numeric_cols[2]}")

            # Visualize with a 3D scatter plot (3 variables)
            fig = px.scatter_3d(df, x=selected_numeric_cols[0], y=selected_numeric_cols[1], z=selected_numeric_cols[2], color=group_col,
                                title=f'{selected_numeric_cols[0]} vs {selected_numeric_cols[1]} vs {selected_numeric_cols[2]} by {group_col}')
            st.plotly_chart(fig)

        else:
            st.write("More than 3 numeric variables selected. Dimensionality reduction required. (You could add PCA here later)")

        # Step 7: Perform t-tests for each selected variable
        st.write("### Performing t-tests for numeric variables:")
        results = []
        for col in selected_numeric_cols:
            group_a = df[df[group_col] == df[group_col].unique()[0]][col]
            group_b = df[df[group_col] == df[group_col].unique()[1]][col]

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

        # Step 8: Display summary table of t-test results
        st.write("### Summary of t-test Results")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)
    else:
        st.error("Please select at least one numeric variable for analysis.")
