import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats

sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 120

# ── Load Data ──────────────────────────────────────────────
df = pd.read_csv('data/healthcare_dataset.csv')
print(df['Billing Amount'].describe().round(2))

# ── Method 1: Z-Score ──────────────────────────────────────
df['Z_Score'] = np.abs(stats.zscore(df['Billing Amount']))
df['Anomaly_ZScore'] = df['Z_Score'] > 3
print(f'\nZ-Score anomalies: {df["Anomaly_ZScore"].sum()}')

# ── Method 2: IQR ──────────────────────────────────────────
Q1, Q3 = df['Billing Amount'].quantile(0.25), df['Billing Amount'].quantile(0.75)
IQR = Q3 - Q1
df['Anomaly_IQR'] = (df['Billing Amount'] < Q1 - 1.5*IQR) | (df['Billing Amount'] > Q3 + 1.5*IQR)
print(f'IQR anomalies   : {df["Anomaly_IQR"].sum()}')

# ── Method 3: Isolation Forest ─────────────────────────────
features = df[['Billing Amount', 'Age', 'Room Number']].copy()
features_scaled = StandardScaler().fit_transform(features)

iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
iso.fit(features_scaled)

df['IF_Label']     = iso.predict(features_scaled)
df['Anomaly_IF']   = df['IF_Label'] == -1
df['Anomaly_Score'] = iso.decision_function(features_scaled)

print(f'Isolation Forest: {df["Anomaly_IF"].sum()}')

# ── Anomaly Score Distribution ─────────────────────────────
plt.figure(figsize=(10, 4))
plt.hist(df['Anomaly_Score'], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
plt.axvline(0, color='red', linestyle='--', label='Decision Boundary')
plt.title('Isolation Forest Anomaly Score Distribution', fontsize=13, fontweight='bold')
plt.xlabel('Anomaly Score (lower = more anomalous)')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.show()

# ── Scatter: Billing vs Age coloured by anomaly ────────────
normal  = df[~df['Anomaly_IF']]
anomaly = df[df['Anomaly_IF']]

plt.figure(figsize=(12, 6))
plt.scatter(normal['Age'],  normal['Billing Amount'],  c='steelblue', alpha=0.4, s=15, label='Normal')
plt.scatter(anomaly['Age'], anomaly['Billing Amount'], c='red', alpha=0.8, s=40,
            marker='X', label=f'Anomaly ({len(anomaly)})')
plt.title('Billing Amount Anomalies — Isolation Forest', fontsize=13, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Billing Amount ($)')
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
plt.legend()
plt.tight_layout()
plt.show()

# ── Distribution Overlay ───────────────────────────────────
plt.figure(figsize=(12, 5))
sns.histplot(normal['Billing Amount'],  bins=40, color='steelblue', alpha=0.7, label='Normal', kde=True)
sns.histplot(anomaly['Billing Amount'], bins=20, color='red',       alpha=0.7, label='Anomaly')
plt.title('Billing Amount — Normal vs Anomalous', fontsize=13, fontweight='bold')
plt.xlabel('Billing Amount ($)')
plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
plt.legend()
plt.tight_layout()
plt.show()

# ── Method Comparison Bar ──────────────────────────────────
method_counts = {
    'Z-Score': df['Anomaly_ZScore'].sum(),
    'IQR':     df['Anomaly_IQR'].sum(),
    'Isolation Forest': df['Anomaly_IF'].sum()
}
plt.figure(figsize=(8, 5))
bars = plt.bar(method_counts.keys(), method_counts.values(),
               color=['coral', 'seagreen', 'steelblue'])
plt.title('Anomalies Detected by Each Method', fontsize=13, fontweight='bold')
plt.ylabel('Count')
for bar, val in zip(bars, method_counts.values()):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             str(val), ha='center', fontweight='bold')
plt.tight_layout()
plt.show()

# ── Top Anomalous Records ──────────────────────────────────
df['Anomaly_Flag'] = df['Anomaly_IF'].map({True: '⚠️ ANOMALY', False: 'Normal'})
top = (df[df['Anomaly_IF']]
       .sort_values('Anomaly_Score')
       [['Name','Age','Medical Condition','Admission Type','Billing Amount','Anomaly_Flag']]
       .head(15))
print('\n=== Top 15 Anomalous Records ===')
pd.options.display.float_format = '{:,.2f}'.format
print(top.to_string(index=False))

# ── Summary Stats ──────────────────────────────────────────
comp = df.groupby('Anomaly_IF')['Billing Amount'].agg(['mean','median','min','max','count'])
comp.index = ['Normal', 'Anomaly']
print('\n=== Billing: Normal vs Anomaly ===')
print(comp.round(2))

# ── Anomaly by Condition ───────────────────────────────────
plt.figure(figsize=(11, 5))
anom_cond = df[df['Anomaly_IF']]['Medical Condition'].value_counts()
sns.barplot(x=anom_cond.index, y=anom_cond.values, palette='Reds_r')
plt.title('Anomalous Records by Medical Condition', fontsize=13, fontweight='bold')
plt.ylabel('Count')
plt.tight_layout()
plt.show()