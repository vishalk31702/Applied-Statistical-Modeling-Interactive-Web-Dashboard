# Statistical Analysis Dashboard

An interactive statistical analysis dashboard built with **Python** and **Streamlit** for Exploratory Data Analysis, Hypothesis Testing, and Statistical Modeling.

## Features

### 1. Dataset Selection
The application supports the following datasets:

- Medical Insurance Costs
- Restaurant Tipping Behavior
- California / Ames Housing Subset

The selected dataset is displayed interactively in the application.

### 2. Data Exploration

The EDA module provides:

- Dataset preview
- Number of rows and columns
- Numerical and categorical column information
- Missing-value information
- Descriptive statistics:
  - Mean
  - Median
  - Standard deviation
  - IQR
  - Skewness
  - Kurtosis
- Box plots
- Histograms / distribution plots
- Scatter plots
- Pair plots
- Correlation matrix

### 3. Data Preprocessing

Optional preprocessing is available:

- Missing-value removal or filling
- Handling columns with more than 50% missing values
- Mean, median, or mode filling
- Label encoding of categorical variables
- Standard scaling
- Min-Max scaling

Encoded categorical columns are excluded from numerical EDA analysis.

### 4. Hypothesis Testing

#### Hypothesis Test 1: Compare Two Groups

The application allows the user to select:

- A categorical variable
- Two groups
- A numerical variable

The application performs:

1. Shapiro-Wilk normality test
2. Levene's test for equality of variance
3. Independent two-sample t-test when assumptions are satisfied
4. Welch's t-test when the groups are normal but have unequal variance
5. Mann-Whitney U test when normality is not satisfied
6. Automatic conclusion using significance level α = 0.05

#### Hypothesis Test 2: Chi-Square Test

The user selects two categorical variables. The application generates:

- Contingency table
- Expected frequencies
- Chi-square statistic
- p-value
- Degrees of freedom
- Statistical conclusion at α = 0.05

#### Hypothesis Test 2: One-Way ANOVA

The user selects:

- A categorical grouping variable with three or more groups
- A numerical variable

The application calculates:

- Group-wise summary statistics
- F-statistic
- p-value
- Statistical conclusion at α = 0.05

### 5. Statistical Modeling

The application implements multiple linear regression using:

`statsmodels.api.OLS`

The user selects:

- Dependent variable (Y)
- One or more independent variables (X)

Categorical predictors are converted to dummy variables using one-hot encoding.

The model provides:

- Regression coefficients
- p-values
- 95% confidence intervals
- R²
- Adjusted R²
- F-statistic
- Overall model p-value
- Automatic coefficient interpretation
- Complete statsmodels OLS summary

### 6. Regression Diagnostics

The application provides diagnostic checks for the fitted regression model.

#### Linearity and Homoscedasticity

A Residuals vs Fitted Values plot is generated to examine:

- Linearity
- Constant variance of residuals

#### Normality of Residuals

A Q-Q plot is generated and the Jarque-Bera test is reported.

The application displays:

- Jarque-Bera statistic
- Jarque-Bera p-value
- Residual skewness
- Residual kurtosis

#### Multicollinearity

Variance Inflation Factor (VIF) is calculated for the model predictors.

A VIF greater than 5 is flagged as a potential multicollinearity issue.

## Project Structure

```text
SM_Assignment/
│
├── demo.py
├── requirements.txt
├── README.md
│
└── data/
    ├── insurance.csv
    ├── tips.csv
    └── housing.csv
```

## Installation

### 1. Clone or download the project

Open a terminal in the project directory.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run demo.py
```

The application will open in your default web browser.

## Required Python Libraries

The application uses:

- Streamlit for the interactive dashboard
- Pandas for data manipulation
- NumPy for numerical operations
- Plotly for interactive visualizations
- Scikit-learn for preprocessing
- SciPy for statistical hypothesis tests
- Statsmodels for OLS regression and statistical diagnostics
- Matplotlib for regression diagnostic plots

## Important Note

The current `demo.py` contains a local Windows path for the Medical Insurance dataset:

```python
"C:\Users\visha\Desktop\Study\New folder\SM_Assignment\insurance.csv"
```

For sharing, GitHub, or cloud deployment, change this to a relative project path such as:

```python
"data/insurance.csv"
```

Make sure the corresponding CSV file is placed inside the `data` folder.

## Statistical Significance

The application uses:

```text
α = 0.05
```

Decision rule:

- p-value < 0.05 → Reject H₀
- p-value ≥ 0.05 → Fail to reject H₀

## Assignment Coverage

This project covers the following assignment components:

- Exploratory Data Analysis
- Descriptive statistical measures
- Data visualization
- Data preprocessing
- Hypothesis Test 1
- Chi-Square Test
- One-Way ANOVA
- Multiple Linear Regression
- OLS coefficient analysis
- R² and Adjusted R²
- Residual diagnostics
- Q-Q plot
- Jarque-Bera normality test
- Variance Inflation Factor (VIF)
- Interactive Streamlit interface

## Author

M.Sc. Data Science — Statistical Modeling with Python
