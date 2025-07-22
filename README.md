# Churn Prediction For SyrialTel Telecom

<img width="589" height="431" alt="customer chrn" src="https://github.com/user-attachments/assets/b5f494f6-041e-4058-b3dc-4944228dfd02" />

## Overview
Telecommunication companies face significant revenue loss due to customer churn. Retaining existing customers is 5–25x more cost-effective than acquiring new ones. However, many businesses struggle to accurately identify which customers are at risk of leaving. 
This project aims to predict customer churn using supervised machine learning algorithms. Churn prediction helps businesses identify customers likely to leave, enabling them to take proactive measures to retain them.
I implemented and evaluated both a Tuned Decision Tree Classifier and a Gradient Boosting Classifier to find the most effective approach.

## Business Understanding

Customer churn is a key metric in subscription-based and telecom businesses. Retaining existing customers is more cost-effective than acquiring new ones. Early churn prediction can help target at-risk customers with interventions to improve retention.

#### Stakeholder

SyriaTel's Marketing and Customer Retention Team 

## Data Understanding

The dataset in use was obtained from <https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset?resource=download>. 
It contains SyriaTel Telecom customers' information and key features such as:
Plan details (e.g., international_plan, voice_mail_plan)

Usage metrics (e.g., total_day_minutes, customer_service_calls)

Account information (e.g., account_length, total_intl_charge)

Target variable: churn_1 (1 = churned, 0 = retained)

This dataset is a historical record of SyriaTel customer base with features that describe customer behaviour, service usage and interactions with the company. It includes both behavioural and service attributes directly associated with satisfaction and retention which allows you to find trends that cause churn.

## Modeling
I used three models:
1. Logistic Regression: used as a baseline model and for handling binary classification for target variable churn
2. Tuned Decision Trees: used to capture more complex features
  Hyperparameters like max_depth, min_samples_split, and criterion were tuned using GridSearchCV.
3. Gradient Boosting Classifier: which captures the most complex patterns in the data.  

## Evaluation
These models' performance were then evaluated using classification metrics like accuracy, ROC-AUC score, precision, recall and F1 score(classification report)
1. Logistic Regression Results
    Accuracy = 75%    ROC-AUC score = 0.82
   
- The logistic regression model correctly predicted 75% of cases with good performance of 0.82.

2. Tuned Decision Trees Results
   Accuracy =  90%   ROC-AUC score = 0.85
  
- The model performed better, reaching 90% accuracy and a score of 0.85.

3. Gradient Boosting Classifier Results
   Accuracy = 92%   ROC-AUC score = 0.92
  
- The model gave the best results, with 92% accuracy and a top performance score of 0.92
  
4 ROC-AUC comparison

<img width="609" height="333" alt="bc8601bd-5340-4a53-b8d7-53f2ec04ff21" src="https://github.com/user-attachments/assets/b26ff012-bc6f-485c-8486-1832d6abe613" />

- The closer the ROC curve is to the top left corner, the better the model’s performance.
  
- Gradient Boosting shows a better trade-off between True Positive Rate (Sensitivity/Recall) and False Positive Rate, making it the superior model for churn prediction.
  
- Logistic Regression, while performing decently, has a lower ability to correctly classify churn cases compared to Random Forest.

## Conclusion
This project demonstrates that churn can be predicted with high accuracy using machine learning.
Gradient Boosting emerged as the best-performing model, and the top churn indicators were:

  - Having an international plan
  
  -  High customer service calls
  
  - High daytime call usage
  
  - Lack of a voicemail plan

These insights can directly inform retention campaigns and improve customer targeting. Future improvements may include:

  - Incorporating time-based features (e.g., tenure trends)
  
  -  Real-time churn prediction
  
   - Model deployment for business use

## Libraries used
Data Handling

- pandas – For data manipulation and preprocessing

- numpy – For numerical operations

📊 Data Visualization
- matplotlib – To create plots and graphs

- seaborn – For enhanced visualizations (heatmaps, distributions, etc.)

📈 Machine Learning Models
- sklearn.model_selection – For train/test split and cross-validation (train_test_split, GridSearchCV)

- sklearn.tree – For building decision tree models

- sklearn.ensemble – For using Gradient Boosting (GradientBoostingClassifier)

- sklearn.metrics – For evaluation metrics (classification_report, confusion_matrix, roc_auc_score, roc_curve)

📊 Model Explanation & Tuning
- sklearn.preprocessing – For label encoding or scaling features

- sklearn.pipeline – To streamline preprocessing and modeling steps
