# Customer Conversion Prediction (ML)

## 📌 Project Overview
This project applies machine learning techniques to predict customer conversions based on ad impressions, engagement metrics, and demographics. The goal is to help advertisers optimize their targeting strategies by identifying potential customers who are likely to convert.

## 🔧 How It Works
- Cleans and preprocesses customer interaction data.
- Encodes categorical variables and scales numerical features.
- Trains multiple machine learning models (Logistic Regression, Random Forest, XGBoost).
- Evaluates model performance using accuracy and AUC-ROC scores.
- Optimizes hyperparameters for the best model.
- Visualizes feature importance.

## 🛠️ Technologies Used
- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Scikit-learn (Machine Learning)
- XGBoost (Boosted Decision Trees)
- Jupyter Notebook (Visualization & Analysis)

## 🚀 Installation & Usage
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/YOUR-USERNAME/customer-conversion-ml.git
cd customer-conversion-ml
```

### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 3️⃣ Run the Machine Learning Script
```sh
python src/customer_conversion_ml.py
```

## 📊 Dataset
The dataset contains information on user interactions with ads, including:
- **Demographics**: Age, Gender, Device Type
- **Ad Engagement**: Time Spent, Number of Impressions, Number of Clicks
- **Target Variable**: Converted (1 = Yes, 0 = No)

Dataset file: `data/customer_conversion_data.csv`

## 🔬 Model Evaluation
- Accuracy and AUC-ROC scores are used to compare models.
- Hyperparameter tuning is applied to optimize Random Forest performance.
- Feature importance is visualized to understand key predictors.

## 🏗️ Future Improvements
- Implement deep learning models for improved prediction.
- Incorporate time-series analysis for ad campaign performance tracking.
- Deploy as a real-time API.


