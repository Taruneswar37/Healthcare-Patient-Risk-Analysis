import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 120

# ── Load & Feature Engineering ─────────────────────────────
df = pd.read_csv('data/healthcare_dataset.csv')
df_model = df.copy()

df_model['Date of Admission'] = pd.to_datetime(df_model['Date of Admission'])
df_model['Discharge Date']    = pd.to_datetime(df_model['Discharge Date'])
df_model['Length of Stay']    = (df_model['Discharge Date'] - df_model['Date of Admission']).dt.days
df_model['Admission Month']   = df_model['Date of Admission'].dt.month
df_model['Admission DayOfWeek'] = df_model['Date of Admission'].dt.dayofweek

df_model.drop(columns=['Name', 'Doctor', 'Hospital', 'Date of Admission', 'Discharge Date'], inplace=True)

# ── Encode Categoricals ────────────────────────────────────
cat_cols = ['Gender', 'Blood Type', 'Medical Condition',
            'Insurance Provider', 'Admission Type', 'Medication']
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le

le_target = LabelEncoder()
df_model['Test Results'] = le_target.fit_transform(df_model['Test Results'])
print(f'Target classes: {le_target.classes_}')

# ── Train / Test Split ─────────────────────────────────────
X = df_model.drop(columns=['Test Results'])
y = df_model['Test Results']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f'Train: {X_train.shape[0]} | Test: {X_test.shape[0]}')

# ── Train Random Forest ────────────────────────────────────
rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                             min_samples_split=5, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f'\nRandom Forest Accuracy: {accuracy_score(y_test, y_pred_rf)*100:.2f}%')

# ── Cross Validation ───────────────────────────────────────
cv = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
print(f'5-Fold CV: {cv.mean():.4f} ± {cv.std():.4f}')

# ── Model Comparison ───────────────────────────────────────
models = {
    'Random Forest':      rf,
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
    'Gradient Boosting':  GradientBoostingClassifier(n_estimators=100, random_state=42)
}
results = {}
for name, model in models.items():
    if name != 'Random Forest':
        model.fit(X_train, y_train)
    results[name] = accuracy_score(y_test, model.predict(X_test))

plt.figure(figsize=(8, 5))
bars = plt.bar(results.keys(), results.values(), color=['steelblue', 'coral', 'seagreen'])
plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for bar, val in zip(bars, results.values()):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.show()

# ── Classification Report ──────────────────────────────────
print('\n=== Classification Report ===')
print(classification_report(y_test, y_pred_rf, target_names=le_target.classes_))

# ── Confusion Matrix ───────────────────────────────────────
cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_target.classes_)
fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(cmap='Blues', ax=ax, colorbar=False)
plt.title('Confusion Matrix — Random Forest', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ── Feature Importance ─────────────────────────────────────
feat_imp = pd.DataFrame({'Feature': X.columns,
                          'Importance': rf.feature_importances_}
                        ).sort_values('Importance', ascending=True)

plt.figure(figsize=(10, 7))
plt.barh(feat_imp['Feature'], feat_imp['Importance'], color='steelblue')
plt.title('Feature Importance — Random Forest', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

# ── Predicted vs Actual ────────────────────────────────────
results_df = pd.DataFrame({
    'Actual':    le_target.inverse_transform(y_test),
    'Predicted': le_target.inverse_transform(y_pred_rf)
})
results_df['Correct'] = results_df['Actual'] == results_df['Predicted']

print('\n=== Sample Predictions (first 20) ===')
print(results_df.head(20).to_string(index=False))
print(f'\nAccuracy: {results_df["Correct"].mean()*100:.2f}%')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Actual vs Predicted Test Results', fontsize=14, fontweight='bold')
sns.barplot(x=results_df['Actual'].value_counts().index,
            y=results_df['Actual'].value_counts().values, ax=axes[0], palette='Set2')
axes[0].set_title('Actual Distribution')
sns.barplot(x=results_df['Predicted'].value_counts().index,
            y=results_df['Predicted'].value_counts().values, ax=axes[1], palette='Set2')
axes[1].set_title('Predicted Distribution')
plt.tight_layout()
plt.show()