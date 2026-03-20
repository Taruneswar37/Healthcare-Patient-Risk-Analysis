import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 120

# ── Load Data ──────────────────────────────────────────────
df = pd.read_csv('data/healthcare_dataset.csv')
print(f'Shape: {df.shape}')
print(df.head())
print(df.describe())
print(df.isnull().sum())

# ── 1. Distributions: Age, Billing Amount, Room Number ─────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Distribution of Numerical Features', fontsize=16, fontweight='bold')

sns.histplot(df['Age'], bins=20, kde=True, ax=axes[0], color='steelblue')
axes[0].set_title('Age Distribution')
axes[0].axvline(df['Age'].mean(), color='red', linestyle='--', label=f"Mean: {df['Age'].mean():.1f}")
axes[0].legend()

sns.histplot(df['Billing Amount'], bins=30, kde=True, ax=axes[1], color='seagreen')
axes[1].set_title('Billing Amount Distribution')
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
axes[1].axvline(df['Billing Amount'].mean(), color='red', linestyle='--',
                label=f"Mean: ${df['Billing Amount'].mean():,.0f}")
axes[1].legend()

sns.histplot(df['Room Number'], bins=20, kde=True, ax=axes[2], color='darkorange')
axes[2].set_title('Room Number Distribution')

plt.tight_layout()
plt.show()

# ── 2. Box Plots ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Boxplots — Outlier Detection', fontsize=14, fontweight='bold')

sns.boxplot(y=df['Age'], ax=axes[0], color='steelblue')
axes[0].set_title('Age')

sns.boxplot(y=df['Billing Amount'], ax=axes[1], color='seagreen')
axes[1].set_title('Billing Amount')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

plt.tight_layout()
plt.show()

# ── 3. Frequency: Medical Condition, Admission Type, Medication ─
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Frequency of Categorical Features', fontsize=16, fontweight='bold')

cat_cols = ['Medical Condition', 'Admission Type', 'Medication']
colors = ['steelblue', 'seagreen', 'coral']

for ax, col, color in zip(axes, cat_cols, colors):
    counts = df[col].value_counts()
    sns.barplot(x=counts.values, y=counts.index, ax=ax, color=color)
    ax.set_title(f'{col}')
    ax.set_xlabel('Count')
    for i, val in enumerate(counts.values):
        ax.text(val + 5, i, str(val), va='center', fontsize=9)

plt.tight_layout()
plt.show()

# ── 4. Pie Charts ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Proportional Distribution', fontsize=15, fontweight='bold')

for ax, col in zip(axes, cat_cols):
    counts = df[col].value_counts()
    ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
           startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    ax.set_title(col)

plt.tight_layout()
plt.show()

# ── 5. Extra Insights ───────────────────────────────────────
plt.figure(figsize=(14, 5))
sns.boxplot(data=df, x='Medical Condition', y='Age', palette='Set2')
plt.title('Age by Medical Condition', fontsize=13, fontweight='bold')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
sns.violinplot(data=df, x='Admission Type', y='Billing Amount', palette='Set3')
plt.title('Billing Amount by Admission Type', fontsize=13, fontweight='bold')
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
plt.tight_layout()
plt.show()

print(df.groupby('Admission Type')['Billing Amount'].describe().round(2))
print(df['Test Results'].value_counts())