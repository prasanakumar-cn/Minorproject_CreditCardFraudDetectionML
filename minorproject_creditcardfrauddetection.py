import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, f1_score, roc_curve
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# LODING DATASET
data = pd.read_csv("creditcard.csv")   # Kaggle dataset
print(data.head())
print(data.info())

# DATA CLEANING AND PREPROCESSING
# Check missing values
print(data.isnull().sum())

# Select useful features (example)
X = data[['Amount','Time']]   # or include location if available
y = data['Class']             # 1 = Fraud, 0 = Normal

# Scale values
scaler = StandardScaler()
X = scaler.fit_transform(X)


#TRAIN TEST_SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)


# HANDLING IMBALENCED DATA -
# Ooversampling Smote
smote = SMOTE(random_state=42)
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)

# undersampling
under = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = under.fit_resample(X_train, y_train)

# TRAIN LOGISTIC REGRESSION
# with oversampled data
model_over = LogisticRegression(max_iter=1000)
model_over.fit(X_train_over, y_train_over)
pred_over = model_over.predict(X_test)
prob_over = model_over.predict_proba(X_test)[:,1]

#with undersampled data
model_under = LogisticRegression(max_iter=1000)
model_under.fit(X_train_under, y_train_under)
pred_under = model_under.predict(X_test)
prob_under = model_under.predict_proba(X_test)[:,1]


# MODEL EVALUATION
# classification report
print("Oversampling Result")
print(classification_report(y_test, pred_over))
print("F1 Score:", f1_score(y_test, pred_over))
print("ROC AUC:", roc_auc_score(y_test, prob_over))

print("Undersampling Result")
print(classification_report(y_test, pred_under))
print("F1 Score:", f1_score(y_test, pred_under))
print("ROC AUC:", roc_auc_score(y_test, prob_under))


# ROC CURVE
fpr, tpr, _ = roc_curve(y_test, prob_over)
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Logistic Regression")
plt.show()







