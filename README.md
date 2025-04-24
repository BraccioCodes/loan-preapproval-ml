# Loan Preapproval Prediction (Logistic Regression + XGBoost)

## 🔎 Project Overview

A classification model to predict whether a loan application should be approved or rejected based on various financial and demographic features. First I tried using logistic regression for a basic model, and then moved onto XGBoost, a state-of-the-arm boosting method. This will be the final project to complete my data analytics certificate, and the motivation behind it was my experience working as a bank teller and gauging people's eligibility for a loan on a daily basis.

---

## 📊 Dataset

The dataset used to train these models comes from Kaggle user architsharma01, and can be accessed [here](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset). It contains categorical as well as quantitative variables that are usually considered when looking at a member's approval rate for a loan application. Some of the features this dataset has are:

- **Number of dependents**
- **Graduate status**
- **Employment status**
- **Annual income**
- **Loan amount**
- **Loan term**
- **Credit score**
- **Assets value**

## 📚 Tools & Libraries

- **Pandas**
- **NumPy**
- **Scikit-learn**
- **XGBoost**
- **Matplotlib / Seaborn**

---

## 🔍 Problem Statement

Lenders want to reduce risk by only approving applicants who are most likely to repay their loans. The goal was to build a classification model that minimizes **false approvals** while still capturing the majority of valid ones.

---

## 🧠 Preprocessing and EDA

I plotted the class balance for our target variable, **loan status**, and found that my dataset was fairly imbalanced, with 2656 records belonging to the positive (**Approved**) class, and 1613 belonging to the negative (**Rejected**) class. I also looked for null values and duplicate values and found none. Outlier detection was performed by creating a function that uses the IQR method, and percentage of outliers in numerical columns was extremely low, so we left them in the dataset. Finally, categorical data was encoded binarily using mapping and quantitative features were scaled using StandardScaler.

## 📌 Feature selection

After a correlation analysis was conducted, it was concluded that by far credit score is the most important feature for our model's prospective prediction power, with a correlation of 77% with loan status. Almost every feature except for loan ID, number of dependents, and graduate status was kept. This decision boils down to the correlation analysis as well as business inteligenge and my interpretation of the lending services provided at my current institution.

---

## 🧩 Model Testing & Evaluation

Training and testing split was 80/20, and it was also stratified to account for the target variable class imbalance. Also cross-validation on both models was run with StratifiedKFold at 5 splits.

**Logistic Regression**

For this model, I picked 1000 iterations, and I achieved 92% accuracy with 22 false positives (FPs) and 41 false negatives (FNs), with a 96% precision score on the positive class. After adjusting the threshold to 0.6 to prioritize precision and get fewer FPs (10) at the expense of more FNs, precision score went up to 98%. I plotted the ROC curve and the ROC-AUC score was 93%.

===

## 🏁 Results

- Logistic Regression Accuracy: **91.3%**
- XGBoost Accuracy: **98.2%**
- XGBoost ROC-AUC: **0.998**
- False Positives reduced through custom thresholding
- Precision optimized for lender safety

---

## ✅ Why This Matters

By adjusting thresholds and using XGBoost, we created a high-performing model that balances **approval accuracy** with **risk control**, making it practical for real-world loan application screening.


