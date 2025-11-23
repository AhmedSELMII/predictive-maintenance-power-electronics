"""
Train machine learning models for failure prediction
Two models:
1. Binary Classifier: Will component fail in next 24 hours?
2. RUL Regressor: How many hours until failure?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    roc_curve,
    mean_squared_error, 
    mean_absolute_error,
    r2_score
)
import joblib
import os

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PREDICTIVE MAINTENANCE ML TRAINING PIPELINE")
print("="*80)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================

print("\n📂 Loading engineered features...")
df = pd.read_csv('data/power_electronics_features.csv')
print(f"   Dataset shape: {df.shape}")
print(f"   Total samples: {len(df):,}")
print(f"   Number of components: {df['component_id'].nunique()}")
print(f"   Failure rate: {df['failure_occurred'].mean()*100:.2f}%")

# ============================================================
# STEP 2: PREPARE FEATURES AND TARGETS
# ============================================================

print("\n🔧 Preparing ML datasets...")

# Exclude non-feature columns
exclude_cols = ['timestamp', 'component_id', 'health_status', 'failure_occurred']
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"   Features available: {len(feature_cols)}")

# Features (X)
X = df[feature_cols]

# Target 1: Binary classification (will fail in next 24 hours?)
print("\n   Creating binary classification target...")
df = df.sort_values(['component_id', 'timestamp']).reset_index(drop=True)

# Look ahead 24 hours: will there be a failure?
df['will_fail_soon'] = df.groupby('component_id')['failure_occurred'].transform(
    lambda x: x.rolling(window=24, min_periods=1).max().shift(-24)
).fillna(0).astype(int)

y_classification = df['will_fail_soon']

print(f"   Classification target created:")
print(f"   - Will fail soon: {y_classification.sum()} samples ({y_classification.mean()*100:.2f}%)")
print(f"   - Will not fail: {(y_classification==0).sum()} samples ({(y_classification==0).mean()*100:.2f}%)")

# Target 2: Remaining Useful Life (hours until failure)
print("\n   Creating RUL regression target...")

def calculate_rul(group):
    """Calculate hours until failure for each sample in a component"""
    failure_indices = group[group['failure_occurred'] == 1].index
    
    if len(failure_indices) == 0:
        # No failure - healthy component, set high RUL
        return pd.Series([1000] * len(group), index=group.index)
    
    first_failure_idx = failure_indices[0]
    rul_values = []
    
    for idx in group.index:
        if idx >= first_failure_idx:
            rul_values.append(0)  # Already failed
        else:
            # Hours until failure
            hours_to_failure = first_failure_idx - idx
            rul_values.append(hours_to_failure)
    
    return pd.Series(rul_values, index=group.index)

df['RUL'] = df.groupby('component_id').apply(calculate_rul).reset_index(level=0, drop=True)
y_regression = df['RUL']

print(f"   RUL target created:")
print(f"   - Mean RUL: {y_regression.mean():.1f} hours")
print(f"   - Median RUL: {y_regression.median():.1f} hours")
print(f"   - Min RUL: {y_regression.min():.1f} hours")
print(f"   - Max RUL: {y_regression.max():.1f} hours")

# ============================================================
# STEP 3: TRAIN/TEST SPLIT
# ============================================================

print("\n📊 Splitting data into train/test sets...")

# Use stratified split for classification to maintain class balance
X_train, X_test, y_class_train, y_class_test, y_rul_train, y_rul_test = train_test_split(
    X, y_classification, y_regression,
    test_size=0.2,
    random_state=42,
    stratify=y_classification
)

print(f"   Training set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Test set: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
print(f"   Train failure rate: {y_class_train.mean()*100:.2f}%")
print(f"   Test failure rate: {y_class_test.mean()*100:.2f}%")

# ============================================================
# STEP 4: TRAIN BINARY CLASSIFIER
# ============================================================

print("\n" + "="*80)
print("🌲 TRAINING FAILURE PREDICTION CLASSIFIER")
print("="*80)

print("\n   Initializing Random Forest Classifier...")
classifier = RandomForestClassifier(
    n_estimators=100,        # Number of trees
    max_depth=15,            # Maximum tree depth (prevent overfitting)
    min_samples_split=10,    # Minimum samples to split a node
    min_samples_leaf=5,      # Minimum samples in leaf node
    random_state=42,
    n_jobs=-1,               # Use all CPU cores
    class_weight='balanced'  # Handle class imbalance
)

print("   Training model...")
classifier.fit(X_train, y_class_train)
print("   ✅ Training complete!")

# Predictions
print("\n   Making predictions...")
y_pred = classifier.predict(X_test)
y_pred_proba = classifier.predict_proba(X_test)[:, 1]

# Evaluate
print("\n📈 CLASSIFICATION MODEL PERFORMANCE:")
print("="*80)
print("\nClassification Report:")
print(classification_report(y_class_test, y_pred, 
                          target_names=['No Failure', 'Failure Soon'],
                          digits=3))

# Confusion Matrix
cm = confusion_matrix(y_class_test, y_pred)
print("\nConfusion Matrix:")
print(f"                 Predicted")
print(f"               No Fail  |  Will Fail")
print(f"Actual  --------------------------------")
print(f"No Fail     {cm[0,0]:6d}    |   {cm[0,1]:6d}")
print(f"Will Fail   {cm[1,0]:6d}    |   {cm[1,1]:6d}")

# Key metrics
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n📊 Key Metrics:")
print(f"   Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"   Precision: {precision:.3f} (When model predicts failure, it's right {precision*100:.1f}% of time)")
print(f"   Recall:    {recall:.3f} (Model catches {recall*100:.1f}% of actual failures)")
print(f"   F1-Score:  {f1:.3f}")

# ROC-AUC
try:
    roc_auc = roc_auc_score(y_class_test, y_pred_proba)
    print(f"   ROC-AUC:   {roc_auc:.3f}")
except:
    print("   ROC-AUC:   Could not calculate")

# Feature Importance
print("\n🎯 TOP 15 MOST IMPORTANT FEATURES:")
print("="*80)
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': classifier.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(15).iterrows():
    print(f"   {row['feature']:50s} {row['importance']:.4f}")

# ============================================================
# STEP 5: TRAIN RUL REGRESSOR
# ============================================================

print("\n" + "="*80)
print("🌲 TRAINING REMAINING USEFUL LIFE (RUL) REGRESSOR")
print("="*80)

print("\n   Initializing Random Forest Regressor...")
regressor = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

print("   Training model...")
regressor.fit(X_train, y_rul_train)
print("   ✅ Training complete!")

# Predictions
print("\n   Making predictions...")
y_rul_pred = regressor.predict(X_test)

# Evaluate
print("\n📈 RUL REGRESSION MODEL PERFORMANCE:")
print("="*80)

rmse = np.sqrt(mean_squared_error(y_rul_test, y_rul_pred))
mae = mean_absolute_error(y_rul_test, y_rul_pred)
r2 = r2_score(y_rul_test, y_rul_pred)

print(f"\n📊 Regression Metrics:")
print(f"   RMSE (Root Mean Squared Error): {rmse:.2f} hours")
print(f"   MAE (Mean Absolute Error):      {mae:.2f} hours")
print(f"   R² Score:                       {r2:.4f}")
print(f"\n   Interpretation:")
print(f"   - On average, predictions are off by {mae:.1f} hours")
print(f"   - Model explains {r2*100:.1f}% of variance in RUL")

# ============================================================
# STEP 6: SAVE MODELS
# ============================================================

print("\n" + "="*80)
print("💾 SAVING TRAINED MODELS")
print("="*80)

# Create models directory
os.makedirs('models', exist_ok=True)
os.makedirs('docs', exist_ok=True)

# Save models
classifier_path = 'models/failure_classifier.pkl'
regressor_path = 'models/rul_regressor.pkl'
feature_names_path = 'models/feature_names.pkl'

joblib.dump(classifier, classifier_path)
joblib.dump(regressor, regressor_path)
joblib.dump(feature_cols, feature_names_path)

print(f"\n   ✅ Classifier saved to: {classifier_path}")
print(f"   ✅ Regressor saved to: {regressor_path}")
print(f"   ✅ Feature names saved to: {feature_names_path}")

# Save feature importance
feature_importance.to_csv('data/feature_importance.csv', index=False)
print(f"   ✅ Feature importance saved to: data/feature_importance.csv")

# ============================================================
# STEP 7: VISUALIZATIONS
# ============================================================

print("\n" + "="*80)
print("📊 CREATING VISUALIZATIONS")
print("="*80)

# Plot 1: Feature Importance
print("\n   Creating feature importance plot...")
plt.figure(figsize=(12, 10))
top_n = 20
top_features = feature_importance.head(top_n)
plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
plt.yticks(range(len(top_features)), top_features['feature'], fontsize=10)
plt.xlabel('Importance Score', fontsize=12)
plt.title(f'Top {top_n} Most Important Features for Failure Prediction', 
          fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('docs/feature_importance.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: docs/feature_importance.png")

# Plot 2: Confusion Matrix
print("   Creating confusion matrix plot...")
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Failure', 'Will Fail'],
            yticklabels=['No Failure', 'Will Fail'])
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.title('Confusion Matrix - Failure Prediction', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('docs/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: docs/confusion_matrix.png")

# Plot 3: ROC Curve
print("   Creating ROC curve...")
try:
    fpr, tpr, thresholds = roc_curve(y_class_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Failure Prediction', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('docs/roc_curve.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: docs/roc_curve.png")
except Exception as e:
    print(f"   ⚠️ Could not create ROC curve: {e}")

# Plot 4: RUL Predictions vs Actual
print("   Creating RUL prediction plot...")
plt.figure(figsize=(10, 6))
plt.scatter(y_rul_test, y_rul_pred, alpha=0.3, s=10)
plt.plot([0, y_rul_test.max()], [0, y_rul_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual RUL (hours)', fontsize=12)
plt.ylabel('Predicted RUL (hours)', fontsize=12)
plt.title(f'RUL Predictions vs Actual (RMSE={rmse:.1f}h, R²={r2:.3f})', 
          fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('docs/rul_predictions.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: docs/rul_predictions.png")

# ============================================================
# STEP 8: SUMMARY
# ============================================================

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)

print(f"""
📊 SUMMARY:

CLASSIFICATION MODEL (Will component fail in 24h?):
   - Accuracy: {accuracy*100:.1f}%
   - Precision: {precision*100:.1f}% (When it says "failure", it's right {precision*100:.1f}% of time)
   - Recall: {recall*100:.1f}% (Catches {recall*100:.1f}% of actual failures)
   - Top predictor: {feature_importance.iloc[0]['feature']}

RUL REGRESSION MODEL (Hours until failure):
   - RMSE: {rmse:.1f} hours
   - MAE: {mae:.1f} hours
   - R² Score: {r2:.3f}
   - Average prediction error: ±{mae:.1f} hours

💾 SAVED FILES:
   - models/failure_classifier.pkl
   - models/rul_regressor.pkl
   - docs/feature_importance.png
   - docs/confusion_matrix.png
   - docs/roc_curve.png
   - docs/rul_predictions.png

🎯 NEXT STEPS:
   1. Review feature importance plot (docs/feature_importance.png)
   2. Check if thermal resistance features are top predictors (validates physics!)
   3. Update your GitHub README with results
   4. Build dashboard for visualization (next phase)
""")

print("="*80)