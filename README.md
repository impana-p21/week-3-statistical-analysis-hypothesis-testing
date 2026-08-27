Week 3 - Statistical Analysis and Hypothesis Testing in Python
Project Overview

This project focuses on statistical analysis and hypothesis testing using Python. The objective is to determine whether customer satisfaction differs significantly between customers who receive discounts and customers who do not receive discounts.

The project demonstrates practical application of descriptive statistics, exploratory data analysis, hypothesis testing, confidence intervals, effect-size analysis, and data visualization.

Research Question

Do customers who receive discounts have significantly different satisfaction scores compared with customers who do not receive discounts?

Hypotheses
Null Hypothesis (H0)

There is no statistically significant difference in mean customer satisfaction between discounted and non-discounted customers.

Alternative Hypothesis (H1)

There is a statistically significant difference in mean customer satisfaction between discounted and non-discounted customers.

Statistical Methods

The project uses:

Descriptive statistics
Independent two-sample Welch t-test
95% confidence interval
Cohen's d effect size
Chi-square test of independence
Cramer's V
Distribution and outlier assessment
Data visualization

The significance level is set to alpha = 0.05.

Technologies
Python
Pandas
NumPy
SciPy
Statsmodels
Matplotlib
Seaborn
Jupyter Notebook
Project Structure
data/
    customer_purchase_data.csv

notebooks/
    week3_statistical_analysis.ipynb

src/
    statistical_analysis.py

visualizations/
    satisfaction_distribution.png
    satisfaction_by_discount.png
    discount_purchase_counts.png
    satisfaction_boxplot.png

report/
    Week_3_Statistical_Analysis_Report.docx

requirements.txt
README.md

Analysis Workflow
Load the customer dataset.
Inspect data types and missing values.
Clean and prepare the dataset.
Calculate descriptive statistics.
Separate customers according to discount status.
Examine distributional assumptions.
Perform Welch's independent two-sample t-test.
Calculate the confidence interval for the mean difference.
Calculate Cohen's d.
Categorize customer satisfaction.
Perform a chi-square test of independence.
Calculate Cramer's V.
Generate visualizations.
Interpret statistical significance and practical significance.
Document the findings in the final Word report.
Interpretation

A p-value below 0.05 provides evidence against the null hypothesis at the 5% significance level. A p-value equal to or greater than 0.05 means that the sample does not provide sufficient evidence to reject the null hypothesis.

Statistical significance should be considered together with the confidence interval and effect size. A statistically significant result does not automatically imply a practically important business effect.

Business Relevance

Understanding whether discounts are associated with customer satisfaction can support pricing, promotion, and customer-retention decisions. If discounted customers demonstrate significantly different satisfaction levels, businesses can investigate whether promotional strategies influence customer experience and purchasing behavior.

Reproducibility

The Python source code is provided in the src directory and is designed to reproduce the statistical analysis and visualizations when the required dataset and Python dependencies are available.

Author

Week 3 Internship Statistical Analysis Project
