
import pandas as pd
import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import joblib

### Dataset loading ###
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  # 0 = malignant, 1 = benign

### data cleaning ###
## check NaN and filled with the mean value
## Handle outliers
# Handle missing values (if any)

df_loading = X.copy()
df_loading['target']=y
#print(df_loading)
nan_in_df = df_loading.isnull().sum().any()
print('Is NaN values in dataset : ', nan_in_df)
if nan_in_df: 
    count_nan = df_loading.isnull().sum()
    print('Number of NaN values present: 
' + str(count_nan))

df_loading.fillna(df_loading.mean(), inplace=True)

# Handle outliers (remove values more than 3 standard deviations from the mean)
df = df_loading[(np.abs(df_loading - df_loading.mean()) <= (3 * df_loading.std())).all(axis=1)]
print(f'{len(df_loading)-len(df)} outlier data points were removed. ')

### Isolate the target column (df['target']) as y
### Split into training and validation sets
### preprocessing numerical data by StandardScaler

pre_X = df.drop(columns=['target'])
# Create the scaler
scaler = StandardScaler()

# Fit and transform only numeric columns
numerical_cols = pre_X.select_dtypes(include='number').columns
X = pre_X.copy()
X[numerical_cols] = scaler.fit_transform(pre_X[numerical_cols])

# Save the fitted scaler for reuse
joblib.dump(scaler, "scaler.pkl")


'''
# Load the saved scaler
scaler = joblib.load("scaler.pkl")

# Transform new data (e.g., new_df)
new_df_scaled = new_df.copy()
new_df_scaled[numerical_cols] = scaler.transform(new_df[numerical_cols])
'''

y = df['target'].astype('int')

#print(X.info())
#print(y.info())
#print(y.value_counts()) #determine if the dataset is balanced or imbalanced


### Split into training and validation sets ###
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Classifiers to compare 
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000,class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42,class_weight='balanced'),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    "LightGBM": LGBMClassifier(random_state=42)
}

# Dictionary to store scores
results = {}

# SMOTE generates synthetic examples for the minority class based on feature space similarities
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Training loop
for name, clf in classifiers.items():
    # Train
    clf.fit(X_train_resampled, y_train_resampled)
    # Predict
    y_pred = clf.predict(X)

    # Score
    acc = accuracy_score(y, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y, y_pred))
    
    results[name] = acc
