import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Load dataset
df = pd.read_csv(r"C:\Users\Joshua.Mahada\Downloads\customer_conversion_data.csv")

# Display basic info
display(df.head())
print(df.info())

# Visualise conversion distribution
plt.figure(figsize=(13,4))
sns.countplot(x='converted', data=df, palette='coolwarm')
plt.title("Conversion Distribution")
plt.show()

# Encoding categorical features
categorical_features = ["gender", "device_type", "ad_channel"]
encoder = OneHotEncoder(drop='first', sparse=False)
categorical_encoded = pd.DataFrame(encoder.fit_transform(df[categorical_features]))
categorical_encoded.columns = encoder.get_feature_names_out()

# Standardising numerical features
numerical_features = ["age", "time_spent", "num_impressions", "num_clicks"]
scaler = StandardScaler()
numerical_scaled = pd.DataFrame(scaler.fit_transform(df[numerical_features]), columns=numerical_features)

# Combine features
X = pd.concat([numerical_scaled, categorical_encoded], axis=1)
y = df["converted"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Feature importance
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(13,5))
feature_importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importances")
plt.show()

print("Exploratory Data Analysis and Model Evaluation Complete!")

