import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

st.set_page_config(page_title="Statistical Analysis Dashboard",page_icon="📊",layout="wide")

st.title("📊 Statistical Analysis Dashboard")

datasets = {
    "Medical Insurance Costs": "data/insurance.csv",
    "Restaurant Tipping Behavior": "data/tips.csv",
    "California / Ames Housing Subset": "data/housing.csv"
}

option = st.selectbox("Choose the dataset:",options=list(datasets.keys()),index=None,placeholder="Select the Dataset")

if "selected_dataset" not in st.session_state:
    st.session_state.selected_dataset = None

if "preprocessing_done" not in st.session_state:
    st.session_state.preprocessing_done = False

if "processed_df" not in st.session_state:
    st.session_state.processed_df = None

if "encoded_columns" not in st.session_state:
    st.session_state.encoded_columns = []

if "scaled_columns" not in st.session_state:
    st.session_state.scaled_columns = []

if "section" not in st.session_state:
    st.session_state.section = None

if option is not None:

    if st.session_state.selected_dataset != option:
        st.session_state.selected_dataset = option
        st.session_state.preprocessing_done = False
        st.session_state.processed_df = None
        st.session_state.encoded_columns = []
        st.session_state.scaled_columns = []
        st.session_state.section = None

    df_original = pd.read_csv(datasets[option])

    st.header(f"{option} Dataset")
    st.subheader("Dataset Preview")

    st.dataframe(df_original,height=400,use_container_width=True)

    original_numeric_columns = df_original.select_dtypes(include="number").columns.tolist()
    original_categorical_columns = df_original.select_dtypes(exclude="number").columns.tolist()

    with st.expander("Dataset Information"):

        col1,col2,col3,col4 = st.columns(4)

        col1.metric("Rows",df_original.shape[0])
        col2.metric("Columns",df_original.shape[1])
        col3.metric("Numerical Columns",len(original_numeric_columns))
        col4.metric("Missing Values",int(df_original.isna().sum().sum()))

        st.write("**Numerical Columns:**")

        if original_numeric_columns:
            st.write(", ".join(original_numeric_columns))
        else:
            st.write("No numerical columns found.")

        st.write("**Categorical Columns:**")

        if original_categorical_columns:
            st.write(", ".join(original_categorical_columns))
        else:
            st.write("No categorical columns found.")

        st.write("**Column Information:**")

        column_info = pd.DataFrame({"Data Type":df_original.dtypes.astype(str),"Missing Values":df_original.isna().sum(),"Missing %":(df_original.isna().mean()*100).round(2)})

        st.dataframe(column_info,use_container_width=True)

    st.subheader("Preprocessing")

    preprocessing_enabled = st.checkbox("Apply preprocessing")

    if preprocessing_enabled:

        actions = st.multiselect("Select preprocessing actions:",["NA Removal / Filling","Label Encoding","Scaling"])

        missing_columns = df_original.columns[df_original.isna().any()].tolist()

        if "NA Removal / Filling" in actions:

            if missing_columns:

                missing_info = pd.DataFrame({"Missing Values":df_original[missing_columns].isna().sum(),"Missing %":(df_original[missing_columns].isna().mean()*100).round(2)})

                st.write("### Missing Value Information")
                st.dataframe(missing_info,use_container_width=True)

                high_missing_columns = [col for col in missing_columns if df_original[col].isna().mean()*100 > 50]
                normal_missing_columns = [col for col in missing_columns if df_original[col].isna().mean()*100 <= 50]

                high_missing_actions = {}
                normal_missing_actions = {}

                if high_missing_columns:

                    st.warning("The following columns contain more than 50% missing values. Choose whether to remove the column or remove rows containing missing values.")

                    for col in high_missing_columns:

                        high_missing_actions[col] = st.radio(f"{col} has {df_original[col].isna().mean()*100:.2f}% missing values:",["Remove Column","Remove Rows with Missing Value"],horizontal=True,key=f"high_missing_{col}")

                if normal_missing_columns:

                    st.write("### Missing Values ≤ 50%")

                    for col in normal_missing_columns:

                        normal_missing_actions[col] = st.radio(f"How should missing values in '{col}' be handled?",["Fill Missing Values","Remove Rows with Missing Value"],horizontal=True,key=f"normal_missing_{col}")

                    fill_columns = [col for col in normal_missing_columns if normal_missing_actions[col] == "Fill Missing Values"]

                    if fill_columns:

                        fill_method = st.selectbox("Select filling method:",["Mean","Median","Mode"])

            else:

                st.success("No missing values found in the dataset.")

        if "Label Encoding" in actions:
            if original_categorical_columns:
                encoding_columns = st.multiselect("Select categorical columns for Label Encoding:",original_categorical_columns)

            else:
                encoding_columns = []
                st.info("No categorical columns found for label encoding.")

        else:
            encoding_columns = []

        if "Scaling" in actions:
            if original_numeric_columns:
                scaling_columns = st.multiselect("Select numerical columns for Scaling:",original_numeric_columns,default=original_numeric_columns)
                scaling_method = st.selectbox("Select scaling method:",["Standard Scaling","Min-Max Scaling"])

            else:
                scaling_columns = []
                scaling_method = None
                st.info("No numerical columns found for scaling.")

        else:
            scaling_columns = []
            scaling_method = None

        if st.button("Apply Preprocessing",type="primary",width="stretch"):

            processed_df = df_original.copy()
            encoded_columns_result = []
            scaled_columns_result = []

            if "NA Removal / Filling" in actions and missing_columns:
                for col in high_missing_columns:
                    if high_missing_actions[col] == "Remove Column":
                        processed_df.drop(columns=[col],inplace=True)

                    elif high_missing_actions[col] == "Remove Rows with Missing Value":
                        processed_df.dropna(subset=[col],inplace=True)

                for col in normal_missing_columns:

                    if normal_missing_actions[col] == "Remove Rows with Missing Value":
                        processed_df.dropna(subset=[col],inplace=True)

                    elif normal_missing_actions[col] == "Fill Missing Values":

                        if fill_method == "Mean":

                            if pd.api.types.is_numeric_dtype(processed_df[col]):
                                processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
                            else:
                                processed_df[col] = processed_df[col].fillna(processed_df[col].mode().iloc[0])

                        elif fill_method == "Median":

                            if pd.api.types.is_numeric_dtype(processed_df[col]):
                                processed_df[col] = processed_df[col].fillna(processed_df[col].median())
                            else:
                                processed_df[col] = processed_df[col].fillna(processed_df[col].mode().iloc[0])

                        elif fill_method == "Mode":

                            if len(processed_df[col].mode()) > 0:
                                processed_df[col] = processed_df[col].fillna(processed_df[col].mode().iloc[0])

            if "Label Encoding" in actions:

                for col in encoding_columns:

                    if col in processed_df.columns:

                        encoder = LabelEncoder()
                        processed_df[col] = encoder.fit_transform(processed_df[col].astype(str))
                        encoded_columns_result.append(col)

            if "Scaling" in actions and scaling_columns:

                valid_scaling_columns = [col for col in scaling_columns if col in processed_df.columns]

                if valid_scaling_columns:

                    if scaling_method == "Standard Scaling":
                        scaler = StandardScaler()
                    else:
                        scaler = MinMaxScaler()

                    processed_df[valid_scaling_columns] = scaler.fit_transform(processed_df[valid_scaling_columns])
                    scaled_columns_result = valid_scaling_columns

            st.session_state.processed_df = processed_df
            st.session_state.preprocessing_done = True
            st.session_state.encoded_columns = encoded_columns_result
            st.session_state.scaled_columns = scaled_columns_result

            st.success("Preprocessing applied successfully.")

    if preprocessing_enabled and st.session_state.preprocessing_done and st.session_state.processed_df is not None:

        df = st.session_state.processed_df

        st.subheader("Preprocessed Dataset")

        st.dataframe(df,height=400,use_container_width=True)

        with st.expander("Preprocessing Summary"):

            st.write("**Label Encoded Columns:**")

            if st.session_state.encoded_columns:
                st.write(", ".join(st.session_state.encoded_columns))
            else:
                st.write("None")

            st.write("**Scaled Columns:**")

            if st.session_state.scaled_columns:
                st.write(", ".join(st.session_state.scaled_columns))
            else:
                st.write("None")

    else:

        df = df_original

    all_numeric_columns = df.select_dtypes(include="number").columns.tolist()

    if st.session_state.preprocessing_done:
        numeric_columns = [col for col in all_numeric_columns if col not in st.session_state.encoded_columns]
    else:
        numeric_columns = all_numeric_columns

    st.divider()

    left,middle,right = st.columns(3)

    if left.button("EDA",width="stretch",type="primary"):
        st.session_state.section = "EDA"

    if middle.button("Hypothesis Testing",width="stretch"):
        st.session_state.section = "Hypothesis Testing"

    if right.button("Statistical Modeling",width="stretch"):
        st.session_state.section = "Statistical Modeling"

    if st.session_state.section == "EDA":

        st.divider()
        st.header("🔎 Exploratory Data Analysis")

        eda_option = st.radio("Choose EDA operation:",["Descriptive Metrics","Plots"],horizontal=True)

        if eda_option == "Descriptive Metrics":

            st.subheader("Descriptive Metrics")

            if len(numeric_columns) == 0:

                st.warning("No numerical columns found in this dataset.")

            else:

                metrics = pd.DataFrame(index=numeric_columns)
                metrics["Mean"] = df[numeric_columns].mean()
                metrics["Median"] = df[numeric_columns].median()
                metrics["Standard Deviation"] = df[numeric_columns].std()
                metrics["IQR"] = df[numeric_columns].quantile(0.75)-df[numeric_columns].quantile(0.25)
                metrics["Skewness"] = df[numeric_columns].skew()
                metrics["Kurtosis"] = df[numeric_columns].kurtosis()

                st.dataframe(metrics.style.format("{:.4f}"),use_container_width=True)

        elif eda_option == "Plots":

            if len(numeric_columns) == 0:

                st.warning("No numerical columns available for plotting.")

            else:

                plot_type = st.selectbox("Choose plot type:",["Box Plot","Histogram / KDE","Scatter Plot"])

                if plot_type == "Box Plot":

                    st.subheader("Box Plot")

                    box_mode = st.radio("Select mode:",["Individual Columns","All Columns"],horizontal=True)

                    if box_mode == "Individual Columns":

                        selected_columns = st.multiselect("Select numerical columns:",numeric_columns)

                        if selected_columns:

                            for column in selected_columns:

                                fig = px.box(df,y=column,points="outliers",title=f"Box Plot - {column}")
                                st.plotly_chart(fig,use_container_width=True)

                        else:

                            st.info("Select one or more numerical columns.")

                    else:

                        fig = px.box(df,y=numeric_columns,points="outliers",title="Box Plot - All Numerical Columns")
                        st.plotly_chart(fig,use_container_width=True)

                elif plot_type == "Histogram / KDE":

                    st.subheader("Distribution Plot")

                    hist_mode = st.radio("Select mode:",["Individual Columns","All Columns"],horizontal=True)

                    if hist_mode == "Individual Columns":

                        selected_columns = st.multiselect("Select numerical columns:",numeric_columns)

                    else:

                        selected_columns = numeric_columns

                    if selected_columns:

                        for column in selected_columns:

                            fig = px.histogram(df,x=column,marginal="box",title=f"Distribution - {column}",histnorm="probability density")
                            st.plotly_chart(fig,use_container_width=True)

                    else:

                        st.info("Select one or more numerical columns.")

                elif plot_type == "Scatter Plot":

                    st.subheader("Scatter Plot")

                    scatter_mode = st.radio("Select mode:",["Two Columns","One vs All","All vs All"],horizontal=True)

                    if scatter_mode == "Two Columns":

                        if len(numeric_columns) < 2:

                            st.info("At least two numerical columns are required.")

                        else:

                            col1,col2 = st.columns(2)

                            x_column = col1.selectbox("Select X-axis:",numeric_columns)
                            y_column = col2.selectbox("Select Y-axis:",numeric_columns,index=1)

                            if x_column == y_column:

                                st.warning("X-axis and Y-axis should be different columns.")

                            else:

                                fig = px.scatter(df,x=x_column,y=y_column,title=f"{x_column} vs {y_column}")
                                st.plotly_chart(fig,use_container_width=True)

                    elif scatter_mode == "One vs All":

                        if len(numeric_columns) < 2:

                            st.info("At least two numerical columns are required for One vs All.")

                        else:

                            selected_column = st.selectbox("Select the main column:",numeric_columns)
                            other_columns = [column for column in numeric_columns if column != selected_column]

                            for column in other_columns:

                                fig = px.scatter(df,x=selected_column,y=column,title=f"{selected_column} vs {column}")
                                st.plotly_chart(fig,use_container_width=True)

                    elif scatter_mode == "All vs All":

                        if len(numeric_columns) < 2:

                            st.info("At least two numerical columns are required for a pair plot.")

                        else:

                            fig = px.scatter_matrix(df,dimensions=numeric_columns,title="Pair Plot - All Numerical Features")
                            fig.update_traces(diagonal_visible=True)
                            st.plotly_chart(fig,use_container_width=True)

                st.divider()

                show_correlation = st.checkbox("Show Correlation Matrix")

                if show_correlation:

                    st.subheader("Correlation Matrix")

                    if len(numeric_columns) < 2:

                        st.info("At least two numerical columns are required for a correlation matrix.")

                    else:

                        correlation = df[numeric_columns].corr()
                        fig = px.imshow(correlation,text_auto=".2f",aspect="auto",title="Numerical Feature Correlation")
                        st.plotly_chart(fig,use_container_width=True)

    elif st.session_state.section == "Hypothesis Testing":

        st.divider()
        st.header("Hypothesis Testing")

        hypothesis_option = st.radio(
            "Select Hypothesis Test:",
            ["Hypothesis Test 1: Compare Two Groups","Hypothesis Test 2: Chi-Square Test","Hypothesis Test 2: One-Way ANOVA"],
            horizontal=True
        )

        if hypothesis_option == "Hypothesis Test 1: Compare Two Groups":

            st.subheader("Hypothesis Test 1: Compare Two Groups")
            st.write("Test whether there is a statistically significant difference between two groups for a numerical variable.")

            available_categorical = [col for col in original_categorical_columns if col not in st.session_state.get("encoded_columns",[])]
            available_numerical = numeric_columns

            if len(available_categorical) == 0:
                st.warning("No categorical variables are available for this test.")
            elif len(available_numerical) == 0:
                st.warning("No numerical variables are available for this test.")
            else:
                categorical_variable = st.selectbox("Select categorical variable:",available_categorical,key="ht1_categorical")

                groups = df[categorical_variable].dropna().unique().tolist()

                if len(groups) < 2:
                    st.warning("The selected categorical variable must contain at least two groups.")
                else:
                    col1,col2,col3 = st.columns(3)

                    group1 = col1.selectbox("Select Group 1:",groups,key="ht1_group1")
                    remaining_groups = [group for group in groups if group != group1]
                    group2 = col2.selectbox("Select Group 2:",remaining_groups,key="ht1_group2")
                    numerical_variable = col3.selectbox("Select numerical variable:",available_numerical,key="ht1_numerical")

                    st.markdown("### Hypotheses")
                    st.write(f"**H₀:** There is no significant difference in {numerical_variable} between {group1} and {group2}.")
                    st.write(f"**H₁:** There is a significant difference in {numerical_variable} between {group1} and {group2}.")
                    st.write("Significance level (α) = 0.05")

                    data_group1 = df.loc[df[categorical_variable] == group1,numerical_variable].dropna()
                    data_group2 = df.loc[df[categorical_variable] == group2,numerical_variable].dropna()

                    if len(data_group1) < 2 or len(data_group2) < 2:
                        st.warning("Both groups must contain at least two valid observations.")
                    else:
                        st.markdown("### Group Summary")

                        summary = pd.DataFrame({
                            "Group":[str(group1),str(group2)],
                            "Count":[len(data_group1),len(data_group2)],
                            "Mean":[data_group1.mean(),data_group2.mean()],
                            "Median":[data_group1.median(),data_group2.median()],
                            "Standard Deviation":[data_group1.std(),data_group2.std()]
                        })

                        st.dataframe(summary,use_container_width=True,hide_index=True)

                        if st.button("Run Hypothesis Test 1",type="primary",width="stretch"):

                            from scipy.stats import shapiro,levene,ttest_ind,mannwhitneyu

                            if len(data_group1) < 3 or len(data_group2) < 3:
                                st.warning("Shapiro-Wilk test requires at least 3 observations in each group.")
                            else:
                                st.markdown("### 1. Shapiro-Wilk Normality Test")

                                shapiro_group1 = shapiro(data_group1)
                                shapiro_group2 = shapiro(data_group2)

                                shapiro_result = pd.DataFrame({
                                    "Group":[str(group1),str(group2)],
                                    "Statistic":[shapiro_group1.statistic,shapiro_group2.statistic],
                                    "p-value":[shapiro_group1.pvalue,shapiro_group2.pvalue]
                                })

                                st.dataframe(shapiro_result,use_container_width=True,hide_index=True)

                                normal_group1 = shapiro_group1.pvalue > 0.05
                                normal_group2 = shapiro_group2.pvalue > 0.05

                                st.markdown("### 2. Levene's Test for Equality of Variances")

                                levene_result = levene(data_group1,data_group2)

                                st.write(f"Levene Statistic: **{levene_result.statistic:.4f}**")
                                st.write(f"p-value: **{levene_result.pvalue:.6f}**")

                                equal_variance = levene_result.pvalue > 0.05

                                if equal_variance:
                                    st.success("Variances can be considered equal because p-value > 0.05.")
                                else:
                                    st.warning("Variances cannot be considered equal because p-value ≤ 0.05.")

                                st.markdown("### 3. Appropriate Statistical Test")

                                if normal_group1 and normal_group2:
                                    if equal_variance:
                                        test_result = ttest_ind(data_group1,data_group2,equal_var=True)
                                        test_name = "Independent Two-Sample t-Test"
                                    else:
                                        test_result = ttest_ind(data_group1,data_group2,equal_var=False)
                                        test_name = "Welch's Two-Sample t-Test"
                                else:
                                    test_result = mannwhitneyu(data_group1,data_group2,alternative="two-sided")
                                    test_name = "Mann-Whitney U Test"

                                st.info(f"Selected Test: **{test_name}**")

                                st.markdown("### 4. Test Result")

                                result_col1,result_col2 = st.columns(2)
                                result_col1.metric("Test Statistic",f"{test_result.statistic:.4f}")
                                result_col2.metric("p-value",f"{test_result.pvalue:.6f}")

                                st.markdown("### 5. Conclusion")

                                if test_result.pvalue < 0.05:
                                    st.error(f"Reject H₀: There is a statistically significant difference in {numerical_variable} between {group1} and {group2}.")
                                else:
                                    st.success(f"Fail to reject H₀: There is no statistically significant difference in {numerical_variable} between {group1} and {group2}.")

        elif hypothesis_option == "Hypothesis Test 2: Chi-Square Test":

            st.subheader("Hypothesis Test 2: Chi-Square Test")
            st.write("Test whether two categorical variables are statistically associated.")

            available_categorical = [col for col in original_categorical_columns if col not in st.session_state.get("encoded_columns",[])]

            if len(available_categorical) < 2:
                st.warning("At least two categorical variables are required for the Chi-Square test.")
            else:
                col1,col2 = st.columns(2)

                categorical_variable1 = col1.selectbox("Select first categorical variable:",available_categorical,key="chi_variable1")
                categorical_variable2 = col2.selectbox("Select second categorical variable:",[col for col in available_categorical if col != categorical_variable1],key="chi_variable2")

                st.markdown("### Hypotheses")
                st.write(f"**H₀:** There is no association between {categorical_variable1} and {categorical_variable2}.")
                st.write(f"**H₁:** There is an association between {categorical_variable1} and {categorical_variable2}.")
                st.write("Significance level (α) = 0.05")

                chi_data = df[[categorical_variable1,categorical_variable2]].dropna()

                if len(chi_data) == 0:
                    st.warning("No valid observations are available for the selected variables.")
                elif chi_data[categorical_variable1].nunique() < 2 or chi_data[categorical_variable2].nunique() < 2:
                    st.warning("Both categorical variables must contain at least two categories.")
                else:
                    contingency_table = pd.crosstab(chi_data[categorical_variable1],chi_data[categorical_variable2])

                    st.markdown("### Contingency Table")
                    st.dataframe(contingency_table,use_container_width=True)

                    if st.button("Run Chi-Square Test",type="primary",width="stretch"):

                        from scipy.stats import chi2_contingency

                        chi2_stat,p_value,dof,expected = chi2_contingency(contingency_table)

                        st.markdown("### Chi-Square Test Result")

                        result_col1,result_col2,result_col3 = st.columns(3)
                        result_col1.metric("Chi-Square Statistic",f"{chi2_stat:.4f}")
                        result_col2.metric("p-value",f"{p_value:.6f}")
                        result_col3.metric("Degrees of Freedom",dof)

                        st.markdown("### Expected Frequencies")
                        expected_table = pd.DataFrame(expected,index=contingency_table.index,columns=contingency_table.columns)
                        st.dataframe(expected_table,use_container_width=True)

                        st.markdown("### Conclusion")

                        if p_value < 0.05:
                            st.error(f"Reject H₀: There is a statistically significant association between {categorical_variable1} and {categorical_variable2}.")
                        else:
                            st.success(f"Fail to reject H₀: There is no statistically significant association between {categorical_variable1} and {categorical_variable2}.")

        elif hypothesis_option == "Hypothesis Test 2: One-Way ANOVA":

            st.subheader("Hypothesis Test 2: One-Way ANOVA")
            st.write("Test whether the mean of a numerical variable differs across three or more groups.")

            available_categorical = [col for col in original_categorical_columns if col not in st.session_state.get("encoded_columns",[])]
            available_numerical = numeric_columns

            if len(available_categorical) == 0:
                st.warning("No categorical variables are available for ANOVA.")
            elif len(available_numerical) == 0:
                st.warning("No numerical variables are available for ANOVA.")
            else:
                col1,col2 = st.columns(2)

                categorical_variable = col1.selectbox("Select grouping variable:",available_categorical,key="anova_categorical")
                numerical_variable = col2.selectbox("Select numerical variable:",available_numerical,key="anova_numerical")

                anova_data = df[[categorical_variable,numerical_variable]].dropna()
                anova_groups = anova_data[categorical_variable].unique().tolist()

                if len(anova_groups) < 3:
                    st.warning("ANOVA requires at least three groups. Please select a categorical variable with three or more groups.")
                else:
                    st.markdown("### Hypotheses")
                    st.write(f"**H₀:** The mean {numerical_variable} is equal across all groups of {categorical_variable}.")
                    st.write(f"**H₁:** At least one group has a significantly different mean {numerical_variable}.")
                    st.write("Significance level (α) = 0.05")

                    st.markdown("### Group Summary")

                    anova_summary = anova_data.groupby(categorical_variable)[numerical_variable].agg(["count","mean","median","std"]).reset_index()
                    anova_summary.columns = ["Group","Count","Mean","Median","Standard Deviation"]

                    st.dataframe(anova_summary,use_container_width=True,hide_index=True)

                    if st.button("Run One-Way ANOVA",type="primary",width="stretch"):

                        from scipy.stats import f_oneway

                        anova_samples = [anova_data.loc[anova_data[categorical_variable] == group,numerical_variable] for group in anova_groups]
                        anova_result = f_oneway(*anova_samples)

                        st.markdown("### ANOVA Result")

                        result_col1,result_col2 = st.columns(2)
                        result_col1.metric("F-Statistic",f"{anova_result.statistic:.4f}")
                        result_col2.metric("p-value",f"{anova_result.pvalue:.6f}")

                        st.markdown("### Conclusion")

                        if anova_result.pvalue < 0.05:
                            st.error(f"Reject H₀: There is a statistically significant difference in mean {numerical_variable} across the groups of {categorical_variable}.")
                        else:
                            st.success(f"Fail to reject H₀: There is no statistically significant difference in mean {numerical_variable} across the groups of {categorical_variable}.") 
                            
                               
    elif st.session_state.section == "Statistical Modeling":

        st.divider()
        st.header("Statistical Modeling")
        st.write("Multiple Linear Regression using Ordinary Least Squares (OLS)")

        available_numerical = numeric_columns
        available_categorical = [col for col in original_categorical_columns if col not in st.session_state.get("encoded_columns",[])]

        if len(available_numerical) < 2:
            st.warning("At least one dependent variable and one independent variable are required for multiple linear regression.")
        else:
            st.subheader("1. Model Formulation")

            dependent_variable = st.selectbox("Select dependent variable (Y):",available_numerical,key="ols_dependent")

            numerical_predictors = [col for col in available_numerical if col != dependent_variable]
            categorical_predictors = [col for col in available_categorical if col != dependent_variable]

            all_predictors = numerical_predictors + categorical_predictors

            if len(all_predictors) == 0:
                st.warning("No independent variables are available.")
            else:
                independent_variables = st.multiselect("Select independent variables (X):",all_predictors,default=numerical_predictors[:min(2,len(numerical_predictors))],key="ols_independent")

                if len(independent_variables) == 0:
                    st.info("Select at least one independent variable to build the regression model.")
                else:
                    st.markdown("### Regression Model")

                    formula_terms = " + ".join(independent_variables)
                    st.code(f"{dependent_variable} = β₀ + β₁X₁ + β₂X₂ + ... + βₖXₖ + ε")

                    st.write(f"**Dependent Variable (Y):** {dependent_variable}")
                    st.write(f"**Independent Variables (X):** {', '.join(independent_variables)}")

                    if st.button("Run OLS Regression",type="primary",width="stretch"):

                        import statsmodels.api as sm
                        import statsmodels.stats.api as sms
                        import matplotlib.pyplot as plt
                        import scipy.stats as stats
                        from statsmodels.stats.outliers_influence import variance_inflation_factor

                        model_columns = [dependent_variable] + independent_variables
                        model_data = df[model_columns].dropna()

                        if len(model_data) <= len(independent_variables) + 1:
                            st.error("There are not enough observations to fit the selected regression model.")
                        else:
                            X = model_data[independent_variables].copy()
                            y = model_data[dependent_variable].copy()

                            categorical_model_columns = [col for col in independent_variables if col in available_categorical]

                            if len(categorical_model_columns) > 0:
                                X = pd.get_dummies(X,columns=categorical_model_columns,drop_first=True,dtype=float)

                            X = X.astype(float)
                            X = sm.add_constant(X)
                            y = pd.to_numeric(y,errors="coerce")

                            valid_rows = X.notna().all(axis=1) & y.notna()
                            X = X.loc[valid_rows]
                            y = y.loc[valid_rows]

                            if X.shape[0] <= X.shape[1]:
                                st.error("There are not enough observations compared with the number of model parameters.")
                            else:
                                model = sm.OLS(y,X).fit()

                                st.subheader("2. OLS Regression Results")

                                st.write("The following model was estimated using `statsmodels.api.OLS`.")

                                result_table = pd.DataFrame({
                                    "Coefficient":model.params,
                                    "P-value":model.pvalues,
                                    "CI Lower 95%":model.conf_int()[0],
                                    "CI Upper 95%":model.conf_int()[1]
                                })

                                result_table.index.name = "Variable"
                                st.dataframe(result_table,use_container_width=True)

                                st.subheader("3. Overall Model Fit")

                                fit_col1,fit_col2,fit_col3,fit_col4 = st.columns(4)

                                fit_col1.metric("R²",f"{model.rsquared:.4f}")
                                fit_col2.metric("Adjusted R²",f"{model.rsquared_adj:.4f}")
                                fit_col3.metric("F-Statistic",f"{model.fvalue:.4f}")
                                fit_col4.metric("Model p-value",f"{model.f_pvalue:.6f}")

                                st.markdown("### Model Interpretation")

                                significant_variables = [variable for variable in model.pvalues.index if variable != "const" and model.pvalues[variable] < 0.05]

                                if len(significant_variables) > 0:
                                    st.success(f"Statistically significant predictors at α = 0.05: {', '.join(significant_variables)}")
                                else:
                                    st.info("No predictor is statistically significant at α = 0.05.")

                                st.write(f"**R² = {model.rsquared:.4f}** means that approximately **{model.rsquared * 100:.2f}%** of the variation in {dependent_variable} is explained by the selected predictors.")

                                st.write(f"**Adjusted R² = {model.rsquared_adj:.4f}** accounts for the number of predictors included in the model.")

                                st.subheader("4. Coefficient Interpretation")

                                interpretation_rows = []

                                for variable in model.params.index:
                                    if variable == "const":
                                        interpretation = f"The intercept represents the expected value of {dependent_variable} when all predictors are zero."
                                    elif model.pvalues[variable] < 0.05:
                                        direction = "increases" if model.params[variable] > 0 else "decreases"
                                        interpretation = f"A one-unit increase in {variable} is associated with a {abs(model.params[variable]):.4f} unit {direction} in {dependent_variable}, holding other predictors constant. The effect is statistically significant."
                                    else:
                                        interpretation = f"The coefficient of {variable} is {model.params[variable]:.4f}, but the effect is not statistically significant at α = 0.05."

                                    interpretation_rows.append([variable,interpretation])

                                interpretation_table = pd.DataFrame(interpretation_rows,columns=["Variable","Interpretation"])
                                st.dataframe(interpretation_table,use_container_width=True,hide_index=True)

                                st.subheader("5. Residual Diagnostics")

                                fitted_values = model.fittedvalues
                                residuals = model.resid

                                st.markdown("### Residuals vs Fitted Values")

                                fig1,ax1 = plt.subplots(figsize=(9,5))
                                ax1.scatter(fitted_values,residuals,alpha=0.6)
                                ax1.axhline(0,linestyle="--")
                                ax1.set_xlabel("Fitted Values")
                                ax1.set_ylabel("Residuals")
                                ax1.set_title("Residuals vs Fitted Values")
                                st.pyplot(fig1)
                                plt.close(fig1)

                                st.write("A random pattern of residuals around zero supports linearity and approximately constant variance. A funnel-shaped or systematic pattern may indicate heteroscedasticity or non-linearity.")

                                st.markdown("### Q-Q Plot of Residuals")

                                fig2,ax2 = plt.subplots(figsize=(9,5))
                                stats.probplot(residuals,dist="norm",plot=ax2)
                                ax2.set_title("Q-Q Plot of Residuals")
                                st.pyplot(fig2)
                                plt.close(fig2)

                                st.write("If the residual points approximately follow the reference line, the residuals are reasonably close to normally distributed.")

                                st.subheader("6. Normality Test of Residuals")

                                jb_test = sms.jarque_bera(residuals)

                                jb_stat = jb_test[0]
                                jb_pvalue = jb_test[1]
                                jb_skewness = jb_test[2]
                                jb_kurtosis = jb_test[3]

                                normality_col1,normality_col2 = st.columns(2)

                                normality_col1.metric("Jarque-Bera Statistic",f"{jb_stat:.4f}")
                                normality_col2.metric("Jarque-Bera p-value",f"{jb_pvalue:.6f}")

                                st.write(f"Residual Skewness: **{jb_skewness:.4f}**")
                                st.write(f"Residual Kurtosis: **{jb_kurtosis:.4f}**")

                                if jb_pvalue > 0.05:
                                    st.success("Fail to reject the normality assumption because the Jarque-Bera p-value is greater than 0.05.")
                                else:
                                    st.warning("Reject the normality assumption because the Jarque-Bera p-value is less than or equal to 0.05.")

                                st.subheader("7. Multicollinearity Check using VIF")

                                vif_data = X.copy()

                                vif_results = []

                                for i,column in enumerate(vif_data.columns):
                                    if column == "const":
                                        continue
                                    vif_value = variance_inflation_factor(vif_data.values,i)
                                    vif_results.append([column,vif_value])

                                vif_table = pd.DataFrame(vif_results,columns=["Variable","VIF"])

                                st.dataframe(vif_table,use_container_width=True,hide_index=True)

                                st.write("VIF values close to 1 indicate little multicollinearity. Larger VIF values indicate stronger multicollinearity among predictors.")

                                high_vif = vif_table[vif_table["VIF"] > 5]

                                if len(high_vif) == 0:
                                    st.success("No serious multicollinearity detected using the VIF > 5 threshold.")
                                else:
                                    st.warning(f"Potential multicollinearity detected in: {', '.join(high_vif['Variable'].tolist())}")

                                st.subheader("8. Complete Statistical Model Summary")

                                with st.expander("View Statsmodels OLS Summary"):
                                    st.text(model.summary())

else:

    st.info("Please select a dataset to continue.")
