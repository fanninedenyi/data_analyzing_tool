import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind

# Title of the app
st.title('Researcher Tool: Guided Variable Analysis with PCA')

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
        st.markdown(f"<div style='border: 2px solid green; padding: 10px;'>"
                    f"You have selected {num_vars} variables, so dimensionality reduction is recommended. We'll use Principal Component Analysis (PCA) for better visualization.</div>", 
                    unsafe_allow_html=True)

        # Step 6: Briefly explain PCA and allow the user to select the number of components
        st.write("#### What is PCA?")
        st.info("PCA is a dimensionality reduction technique that transforms the data into fewer dimensions while retaining most of the important information (variance).")

        # Let the user choose how many components to use
        num_components = st.slider("Select the number of PCA components (2 or 3)", min_value=2, max_value=3, value=2)

        # Step 7: Perform PCA and evaluate variance explained
        pca = PCA(n_components=num_components)
        pca_result = pca.fit_transform(df[selected_numeric_cols])
        explained_variance = pca.explained_variance_ratio_

        st.write(f"#### PCA Variance Explained with {num_components} components:")
        st.write(f"Total variance explained: {explained_variance.sum() * 100:.2f}%")

        if num_components == 2:
            st.write(f"Variance explained by PC1: {explained_variance[0] * 100:.2f}%")
            st.write(f"Variance explained by PC2: {explained_variance[1] * 100:.2f}%")
        elif num_components == 3:
            st.write(f"Variance explained by PC1: {explained_variance[0] * 100:.2f}%")
            st.write(f"Variance explained by PC2: {explained_variance[1] * 100:.2f}%")
            st.write(f"Variance explained by PC3: {explained_variance[2] * 100:.2f}%")

        # Step 8: Visualize PCA results (2D or 3D plot)
        if num_components == 2:
            st.write("### 2D PCA Cluster Plot")
            pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
            pca_df[group_col] = df[group_col]
            fig = px.scatter(pca_df, x='PC1', y='PC2', color=group_col, title="2D PCA Cluster Plot")
            st.plotly_chart(fig)

        elif num_components == 3:
            st.write("### 3D PCA Cluster Plot")
            pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2', 'PC3'])
            pca_df[group_col] = df[group_col]
            fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', color=group_col, title="3D PCA Cluster Plot")
            st.plotly_chart(fig)

        st.markdown("<div style='border: 2px solid blue; padding: 10px;'>"
                    "This plot visualizes the selected variables in reduced dimensions, with different colors for each group.</div>", 
                    unsafe_allow_html=True)

        # Step 9: Perform t-tests for each selected variable
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

        # Step 10: Display summary table of t-test results
        st.write("### Summary of t-test Results")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)
