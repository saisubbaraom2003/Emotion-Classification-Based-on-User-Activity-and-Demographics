# Emotion Classification Based on User Activity and Demographics

## About the Project
This project focuses on predicting the **dominant emotion** of users on a platform using features derived from user activity (such as messages sent, posts per day, and usage time) and demographic data (like age, gender, and platform). The task is treated as a **multi-class classification problem**, where the target variable is the user's dominant emotion.

---

## Techniques Used

### 🧹 Data Preprocessing & Cleaning
- Handling incorrect data types (e.g., converting `Age` from object to numeric).
- Filling missing values using:
  - **Mean** imputation for numeric columns.
  - **Mode** for categorical columns (e.g., `Gender`).
- Dropping records with negligible missing data (<1%).
- **Outlier treatment** using the **IQR (Interquartile Range)** method.

### 📊 Exploratory Data Analysis (EDA)
- Visualizations of **categorical and numeric data distributions** by dominant emotion.
- **Correlation matrix analysis** to identify and remove highly correlated features:
  - `Likes_Received_Per_Day`, `Comments_Received_Per_Day`, `Messages_Sent_Per_Day` removed due to high correlation.

### 🔄 Feature Engineering
- Segregating features into:
  - **Categorical**
  - **Numerical**
  - **Target**
- Encoding:
  - **Label Encoding** for target variable (emotion levels).
  - **One-Hot Encoding** for categorical variables (`Gender`, `Platform`).
- **Feature Scaling** for numerical features (required for distance-based models like KNN, SVC).

---

## 🤖 Machine Learning Models
Several models were trained and compared:

- **Logistic Regression** – Baseline model, fast, interpretable.
- **Decision Tree Classifier** – Non-linear, interpretable.
- **Random Forest Classifier** – Ensemble learning to reduce overfitting.
- **XGBoost Classifier** – Boosting technique, very effective for structured data.
- **K-Nearest Neighbors (KNN)** – Instance-based, sensitive to scaling.
- **Support Vector Classifier (SVC)** – Good for small to medium datasets.

---

## 🧰 Technologies Used

- **Python**
- **Pandas**, **NumPy** – Data manipulation
- **Matplotlib**, **Seaborn** – Visualization
- **Scikit-learn** – Preprocessing, model training, evaluation
- **XGBoost** – Advanced gradient boosting

---

## ☁️ Deployment

- Initially deployed on **Microsoft Azure** and **Amazon Web Services (AWS)** during development and testing phases.
- Due to expiration of free trial credits on those platforms, the final deployment was successfully completed using **[Render](https://render.com/)**, a cloud platform for hosting web applications and APIs.

---

## 🧭 Approach

1. **Data Preparation:** Clean and transform the raw data.
2. **Feature Analysis:** Use EDA and correlation to understand and refine features.
3. **Encoding & Scaling:** Make data suitable for ML models.
4. **Model Training:** Try various algorithms to find the best-performing one.
5. **Model Evaluation:** Compare models using appropriate metrics (not detailed in preview but likely included in notebook).
6. **Deployment:** Deploy the final model using cloud services, ending with Render for continuous access.
