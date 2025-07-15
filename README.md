# 📈 Orange Customer Analysis

This project focuses on analyzing and modeling Orange telecom customer data to understand user behavior, identify churn risks, and recommend upselling/downselling strategies using machine learning. A complete data science pipeline is implemented, from raw data handling to interactive prediction using Streamlit.

---

## 🗂️ Project Structure

1. **Raw Data (Orange Customers Data)**  
   - The original customer dataset provided by Orange is stored in the `data/raw/` directory.

2. **Data Cleaning**  
   - Performed handling of missing values, outliers, data types, and formatting inconsistencies.
   - Stored cleaned data in `data/processed/`.

3. **Data Visualization**  
   - Univariate, bivariate, and multivariate visualizations created using:
     - Matplotlib, Seaborn, Plotly
     - Custom plots for customer activity, service usage, contract types, etc.

4. **Correlation Analysis**  
   - Identified relationships between key features and customer churn or product uptake.
   - Heatmaps and pairplots used to visualize correlation structure.

5. **Feature Engineering**  
   - Created new features such as:
     - Tenure buckets
     - Engagement scores
     - Service combination flags
   - Applied one-hot encoding, label encoding, scaling, and transformation as needed.

6. **Benchmarking ML Models**  
   - Trained and evaluated multiple classification models:
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - XGBoost
     - Gradient Boosting
     - Support Vector Machine (SVM)
   - Used cross-validation and stratified sampling to ensure robustness.

7. **Hyperparameter Tuning**  
   - Applied GridSearchCV and RandomizedSearchCV for tuning:
     - Max depth, learning rate, number of estimators, etc.
   - Metrics used: Accuracy, Precision, Recall, F1-score, AUC-ROC

8. **Pattern Identification & Insights**  
   - Extracted patterns such as:
     - Services linked to higher churn
     - Feature combinations that predict upselling potential
     - Behavior patterns of loyal vs. leaving customers

9. **Streamlit App: Upsell/Downsell Recommender**  
   - Developed an interactive web app using Streamlit:
     - Users can input customer data
     - Get predictions for churn probability
     - Get recommendation: **Upsell**, **Downsell**, or **Retain with offer**
     - Enables Orange to act proactively to retain valuable customers

---
## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/nithyaraaj21/orange-customer-analytics.git
   cd orange-customer-analytics
