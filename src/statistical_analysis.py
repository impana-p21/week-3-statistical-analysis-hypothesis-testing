import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from statsmodels.stats.proportion import proportion_confint


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_PATH = "data/customer_purchase_data.csv"
OUTPUT_DIR = "visualizations"

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())


# ---------------------------------------------------------
# Basic validation
# ---------------------------------------------------------

required_columns = [
    "customer_id",
    "discount_received",
    "satisfaction_score"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ---------------------------------------------------------
# Data preparation
# ---------------------------------------------------------

df = df.dropna(
    subset=[
        "discount_received",
        "satisfaction_score"
    ]
).copy()

df["discount_received"] = df["discount_received"].astype(int)

df["discount_group"] = np.where(
    df["discount_received"] == 1,
    "Discount",
    "No Discount"
)


# ---------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------

print("\nOverall descriptive statistics:")
print(df["satisfaction_score"].describe())

print("\nDescriptive statistics by discount status:")

group_summary = (
    df.groupby("discount_group")["satisfaction_score"]
    .agg(
        count="count",
        mean="mean",
        median="median",
        std="std",
        minimum="min",
        maximum="max"
    )
)

print(group_summary)


# ---------------------------------------------------------
# Group separation
# ---------------------------------------------------------

discount_group = df.loc[
    df["discount_received"] == 1,
    "satisfaction_score"
]

no_discount_group = df.loc[
    df["discount_received"] == 0,
    "satisfaction_score"
]


# ---------------------------------------------------------
# Normality assessment
# ---------------------------------------------------------

print("\nShapiro-Wilk normality tests:")

if len(discount_group) >= 3:
    shapiro_discount = stats.shapiro(discount_group)
    print(
        f"Discount group: "
        f"W={shapiro_discount.statistic:.4f}, "
        f"p={shapiro_discount.pvalue:.4f}"
    )

if len(no_discount_group) >= 3:
    shapiro_no_discount = stats.shapiro(no_discount_group)
    print(
        f"No Discount group: "
        f"W={shapiro_no_discount.statistic:.4f}, "
        f"p={shapiro_no_discount.pvalue:.4f}"
    )


# ---------------------------------------------------------
# Independent two-sample Welch t-test
# ---------------------------------------------------------

t_test = stats.ttest_ind(
    discount_group,
    no_discount_group,
    equal_var=False
)

print("\nWelch independent two-sample t-test:")
print(f"t-statistic = {t_test.statistic:.4f}")
print(f"p-value = {t_test.pvalue:.6f}")


# ---------------------------------------------------------
# Mean difference and confidence interval
# ---------------------------------------------------------

mean_discount = discount_group.mean()
mean_no_discount = no_discount_group.mean()

mean_difference = mean_discount - mean_no_discount

n1 = len(discount_group)
n2 = len(no_discount_group)

var1 = discount_group.var(ddof=1)
var2 = no_discount_group.var(ddof=1)

standard_error = np.sqrt(
    (var1 / n1) + (var2 / n2)
)

df_welch = (
    ((var1 / n1) + (var2 / n2)) ** 2
    /
    (
        ((var1 / n1) ** 2 / (n1 - 1))
        +
        ((var2 / n2) ** 2 / (n2 - 1))
    )
)

critical_value = stats.t.ppf(
    0.975,
    df_welch
)

ci_lower = (
    mean_difference
    - critical_value * standard_error
)

ci_upper = (
    mean_difference
    + critical_value * standard_error
)

print("\nMean difference:")
print(f"{mean_difference:.4f}")

print("\n95% confidence interval:")
print(f"Lower = {ci_lower:.4f}")
print(f"Upper = {ci_upper:.4f}")


# ---------------------------------------------------------
# Cohen's d
# ---------------------------------------------------------

pooled_sd = np.sqrt(
    (
        (n1 - 1) * var1
        +
        (n2 - 1) * var2
    )
    /
    (n1 + n2 - 2)
)

cohens_d = mean_difference / pooled_sd

print("\nCohen's d:")
print(f"{cohens_d:.4f}")


# ---------------------------------------------------------
# Satisfaction categories
# ---------------------------------------------------------

def satisfaction_category(score):
    if score <= 2:
        return "Low"
    elif score <= 4:
        return "Medium"
    else:
        return "High"


df["satisfaction_category"] = (
    df["satisfaction_score"]
    .apply(satisfaction_category)
)


# ---------------------------------------------------------
# Chi-square test
# ---------------------------------------------------------

contingency_table = pd.crosstab(
    df["discount_group"],
    df["satisfaction_category"]
)

print("\nContingency table:")
print(contingency_table)

chi2, p_value, degrees_freedom, expected = (
    stats.chi2_contingency(contingency_table)
)

print("\nChi-square test:")
print(f"Chi-square = {chi2:.4f}")
print(f"Degrees of freedom = {degrees_freedom}")
print(f"p-value = {p_value:.6f}")

expected_table = pd.DataFrame(
    expected,
    index=contingency_table.index,
    columns=contingency_table.columns
)

print("\nExpected frequencies:")
print(expected_table)


# ---------------------------------------------------------
# Cramer's V
# ---------------------------------------------------------

n = contingency_table.values.sum()

min_dimension = min(
    contingency_table.shape
) - 1

cramers_v = np.sqrt(
    chi2 / (n * min_dimension)
)

print("\nCramer's V:")
print(f"{cramers_v:.4f}")


# ---------------------------------------------------------
# Hypothesis decision
# ---------------------------------------------------------

alpha = 0.05

print("\nHypothesis decision:")

if t_test.pvalue < alpha:
    print(
        "Reject H0: there is statistically significant "
        "evidence of a difference in mean satisfaction."
    )
else:
    print(
        "Fail to reject H0: there is insufficient "
        "evidence of a difference in mean satisfaction."
    )


# ---------------------------------------------------------
# Visualization 1: Satisfaction distribution
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.histplot(
    data=df,
    x="satisfaction_score",
    hue="discount_group",
    kde=True,
    bins=10,
    element="step"
)

plt.title(
    "Distribution of Customer Satisfaction Scores"
)

plt.xlabel("Satisfaction Score")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/satisfaction_distribution.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# Visualization 2: Boxplot
# ---------------------------------------------------------

plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="discount_group",
    y="satisfaction_score"
)

plt.title(
    "Customer Satisfaction by Discount Status"
)

plt.xlabel("Customer Group")
plt.ylabel("Satisfaction Score")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/satisfaction_boxplot.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# Visualization 3: Mean satisfaction
# ---------------------------------------------------------

plt.figure(figsize=(9, 6))

sns.barplot(
    data=df,
    x="discount_group",
    y="satisfaction_score",
    errorbar="ci"
)

plt.title(
    "Average Customer Satisfaction by Discount Status"
)

plt.xlabel("Customer Group")
plt.ylabel("Mean Satisfaction Score")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/satisfaction_by_discount.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# Visualization 4: Satisfaction categories
# ---------------------------------------------------------

plt.figure(figsize=(9, 6))

sns.countplot(
    data=df,
    x="satisfaction_category",
    hue="discount_group"
)

plt.title(
    "Satisfaction Categories by Discount Status"
)

plt.xlabel("Satisfaction Category")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/discount_purchase_counts.png",
    dpi=300
)

plt.close()


print("\nAnalysis completed.")
print("Visualization files saved to:", OUTPUT_DIR)
