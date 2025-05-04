# Emotion Classification Based on User Activity and Demographics

## 🎯 About the Project

This project focuses on predicting the **dominant emotion** of users on a platform using features derived from user activity (such as messages sent, posts per day, and usage time) and demographic data (like age, gender, and platform). The task is a **multi-class classification problem**, where the target variable is the user's dominant emotion.

---

## 🧹 Data Preprocessing & Cleaning

- Converted incorrect data types (e.g., `Age` from object to numeric).
- Filled missing values using:
  - **Mean** for numeric columns.
  - **Mode** for categorical columns like `Gender`.
- Dropped records with negligible missing data (<1%).
- **Outlier detection and removal** using the **IQR (Interquartile Range)** method.

---

## 📊 Exploratory Data Analysis (EDA)

- Plotted distributions of categorical and numerical features based on emotions.
- Used a **correlation matrix** to identify and remove multicollinearity:
  - Removed `Likes_Received_Per_Day`, `Comments_Received_Per_Day`, and `Messages_Sent_Per_Day`.

---

## 🔄 Feature Engineering

- Segregated features into:
  - **Categorical**
  - **Numerical**
  - **Target**
- **Label Encoding**: Used for the target emotion column.
- **One-Hot Encoding**: Applied on `Gender`, `Platform`.
- **Feature Scaling**: Standardized numerical features (important for models like KNN and SVC).

---

## 🤖 Machine Learning Models Used

| Model                  | Type                | Comment                                                                 |
|------------------------|---------------------|-------------------------------------------------------------------------|
| LogisticRegression     | Linear              | Simple, fast, good baseline model.                                      |
| DecisionTreeClassifier | Tree-Based          | Easy to interpret, prone to overfitting.                                |
| RandomForestClassifier | Ensemble (Trees)    | Reduces overfitting using bagging.                                      |
| XGBClassifier          | Gradient Boosting   | Powerful and flexible.                                                  |
| KNeighborsClassifier   | Distance-Based      | Needs scaling; good for small data.                                     |
| SVC                    | Kernel-Based        | Effective for complex boundaries.                                       |
| GaussianNB             | Probabilistic       | Very fast; assumes feature independence.                                |
| MLPClassifier          | Neural Network      | Learns complex patterns. Needs proper tuning.                           |
| ExtraTreesClassifier   | Ensemble (Random)   | Like RF, but introduces more randomness for potentially better results. |
| AdaBoostClassifier     | Boosting            | Reduces bias by combining weak learners.                                |
| VotingClassifier       | Ensemble            | Aggregates predictions from multiple models (soft voting used).         |

---

## 🎯 Hyperparameter Tuning

Used **GridSearchCV** for optimizing RandomForestClassifier:

**Parameter Grid:**

- `n_estimators`: [100, 200]
- `max_depth`: [None, 10, 20]
- `min_samples_split`: [2, 5]
- `min_samples_leaf`: [1, 2]
- `bootstrap`: [True, False]

**Results:**

- **Best Parameters**:  
  `{'bootstrap': True, 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}`
- **Best Cross-Validation Score**: 0.99
- **Training Accuracy**: 1.00  
- **Testing Accuracy**: 0.988  

---

## 📈 Evaluation

- **Precision**: Very few false positives.
- **Recall**: Most actual instances are correctly identified.
- **F1-Score**: Balanced between precision and recall.
- **Macro Avg / Weighted Avg**: High and consistent across classes.

### 📊 Classification Report Summary

| Emotion   | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Happiness | 1.00      | 1.00   | 1.00     | 33      |
| Neutral   | 1.00      | 1.00   | 1.00     | 42      |
| Boredom   | 0.94      | 0.97   | 0.96     | 35      |
| Anxiety   | 1.00      | 1.00   | 1.00     | 50      |
| Anger     | 0.98      | 1.00   | 0.99     | 50      |
| Sadness   | 1.00      | 0.95   | 0.97     | 40      |
| **Overall Accuracy** |       |        | **0.99** | **250** |

---

## 📉 Confusion Matrix

Visualized using a heatmap:

| Emotion     | Misclassifications                          |
|-------------|---------------------------------------------|
| Happiness   | 0                                            |
| Neutral     | 0                                            |
| Boredom     | 1 sample misclassified as Anger             |
| Anxiety     | 0                                            |
| Anger       | 0                                            |
| Sadness     | 2 samples misclassified as Boredom          |

> Conclusion: No emotion is consistently misclassified. Minor confusion only between Boredom and Sadness — indicating strong generalization.

---

## ☁️ Deployment

- Initially deployed to:
  - **Microsoft Azure**
  - **Amazon Web Services (AWS)**  
  *(during free trial periods)*

- **Final Deployment**:
  - Moved to **[Render](https://emotion-classification-based-on-user-2gpl.onrender.com/)** after cloud credits expired.
  - Hosted as a **REST API** or **web interface** for real-time predictions.

---

## 📦 Technologies Used

- **Python**
- **Pandas**, **NumPy** – Data preprocessing & manipulation
- **Matplotlib**, **Seaborn** – Data visualization
- **Scikit-learn** – ML modeling and evaluation
- **XGBoost** – Boosting model
- **Render** – Final cloud deployment platform

---

## 🧭 Approach Summary

1. **Data Preparation**: Cleaned and structured the dataset.
2. **EDA**: Visualized feature distributions and relationships.
3. **Feature Engineering**: Encoded and scaled data appropriately.
4. **Model Selection**: Trained multiple models to identify the best.
5. **Hyperparameter Tuning**: Used GridSearchCV for optimization.
6. **Evaluation**: Measured performance using metrics and cross-validation.
7. **Deployment**: Hosted the final model in the cloud using Render.

---

## 🏷️ Emotion Label Mapping

| Emotion   | Label |
|-----------|-------|
| Happiness | 0     |
| Neutral   | 1     |
| Boredom   | 2     |
| Anxiety   | 3     |
| Anger     | 4     |
| Sadness   | 5     |

---

## Contribution
Feel free to contribute by opening issues or submitting pull requests.
## Contact
For any queries, contact me sai.subbu.in@gmail.com

---
**Author:** Sai Subba Rao Mahendrakar  
**Date:** 4 May 2025  
