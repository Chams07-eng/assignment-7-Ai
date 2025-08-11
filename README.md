# assignment-7-Ai
Ethics

# compas_audit.py
# Audit COMPAS recidivism dataset using IBM AIF360
# Requirements: aif360, pandas, numpy, scikit-learn, matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aif360.datasets import CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.postprocessing import RejectOptionClassification
from aif360.algorithms.inprocessing import AdversarialDebiasing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

# 1. Load COMPAS dataset from aif360
dataset = CompasDataset()  # defaults to ProPublica COMPAS processed dataset

# Inspect
print(dataset.features.shape)
print("Protected attribute names:", dataset.protected_attribute_names)
print("Label name:", dataset.label_names)

# Protected attribute: 'race' with privileged group 'Caucasian'
privileged_groups = [{'race': 1}]  # aif360 encodes privileged as 1 (Caucasian)
unprivileged_groups = [{'race': 0}]  # non-Caucasian

# 2. Train-test split
train, test = dataset.split([0.7], shuffle=True)

# 3. Baseline metrics
print("=== Baseline dataset metrics (train) ===")
metric_train = BinaryLabelDatasetMetric(train, unprivileged_groups=unprivileged_groups,
                                       privileged_groups=privileged_groups)
print("Mean label (positive rate):", metric_train.base_rate())
print("Disparate impact (p(unpriv)/p(priv)):", metric_train.disparate_impact())
print("Statistical parity difference:", metric_train.statistical_parity_difference())

# Train a simple classifier (Logistic Regression) on original features
X_train = train.features
y_train = train.labels.ravel()
X_test = test.features
y_test = test.labels.ravel()

scaler = StandardScaler()
clf = LogisticRegression(max_iter=1000)

# Fit on numeric features only (aif360 dataset prepared)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf.fit(X_train_scaled, y_train)

# Create predicted dataset for test
y_pred = clf.predict(X_test_scaled)
test_pred = test.copy()
test_pred.labels = y_pred.reshape(-1,1)

# Compute classification metrics disaggregated by race
class_metric = ClassificationMetric(test, test_pred,
                                    unprivileged_groups=unprivileged_groups,
                                    privileged_groups=privileged_groups)

print("\n=== Classification metrics on test ===")
print("Accuracy:", class_metric.accuracy())
print("False positive rate (unprivileged):", class_metric.false_positive_rate(unprivileged=True))
print("False positive rate (privileged):", class_metric.false_positive_rate(privileged=True))
print("False negative rate (unprivileged):", class_metric.false_negative_rate(unprivileged=True))
print("False negative rate (privileged):", class_metric.false_negative_rate(privileged=True))
print("Disparate impact (predicted):", class_metric.disparate_impact())

# 4. Visualize disparity in FPR between groups
fpr_unpriv = class_metric.false_positive_rate(unprivileged=True)
fpr_priv = class_metric.false_positive_rate(privileged=True)

groups = ['Unprivileged (non-Caucasian)', 'Privileged (Caucasian)']
fpr_vals = [fpr_unpriv, fpr_priv]

plt.figure(figsize=(6,4))
plt.bar(groups, fpr_vals)
plt.title('False Positive Rate by Race Group')
plt.ylabel('False Positive Rate')
plt.ylim(0, max(fpr_vals)*1.2)
for i,v in enumerate(fpr_vals):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
plt.tight_layout()
plt.show()

# 5. Apply a preprocessing mitigation: Reweighing
rw = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
train_rw = rw.fit_transform(train)

# Train classifier on reweighted data (weights in instance attribute)
clf2 = LogisticRegression(max_iter=1000)
clf2.fit(scaler.fit_transform(train_rw.features), train_rw.labels.ravel(), sample_weight=train_rw.instance_weights.ravel())

# Predict on test
y_pred2 = clf2.predict(scaler.transform(test.features))
test_pred2 = test.copy()
test_pred2.labels = y_pred2.reshape(-1,1)

class_metric2 = ClassificationMetric(test, test_pred2,
                                     unprivileged_groups=unprivileged_groups,
                                     privileged_groups=privileged_groups)

print("\n=== After Reweighing (train) & Classifier ===")
print("Accuracy:", class_metric2.accuracy())
print("FPR (unprivileged):", class_metric2.false_positive_rate(unprivileged=True))
print("FPR (privileged):", class_metric2.false_positive_rate(privileged=True))
print("Statistical parity difference (predicted):", class_metric2.statistical_parity_difference())
print("Disparate impact (predicted):", class_metric2.disparate_impact())

# 6. Post-processing: Reject Option Classification (example)
roc = RejectOptionClassification(unprivileged_groups=unprivileged_groups,
                                 privileged_groups=privileged_groups,
                                 low_class_thresh=0.01, high_class_thresh=0.99,
                                 num_class_thresh=50, num_ROC_margin=50)

# Need probabilistic scores to apply ROC; use predict_proba
scores = clf.predict_proba(scaler.transform(test.features))[:,1]
test_score = test.copy()
test_score.scores = scores.reshape(-1,1)

# fit ROC on validation data if required — here we use test as demonstration (in real audit use separate val set)
try:
    roc = roc.fit(test, test_score)
    test_pred_roc = roc.predict(test_score)
    class_metric_roc = ClassificationMetric(test, test_pred_roc,
                                            unprivileged_groups=unprivileged_groups,
                                            privileged_groups=privileged_groups)
    print("\n=== After Reject Option Classification ===")
    print("FPR (unprivileged):", class_metric_roc.false_positive_rate(unprivileged=True))
    print("FPR (privileged):", class_metric_roc.false_positive_rate(privileged=True))
    print("Disparate impact (predicted):", class_metric_roc.disparate_impact())
except Exception as e:
    print("RejectOptionClassification step could not be completed in this run:", e)

# 7. Summary printout of key metrics
print("\n=== Summary ===")
print("Baseline disparate impact (train):", metric_train.disparate_impact())
print("Predicted disparate impact (baseline classifier):", class_metric.disparate_impact())
print("Predicted disparate impact (reweighing):", class_metric2.disparate_impact())
