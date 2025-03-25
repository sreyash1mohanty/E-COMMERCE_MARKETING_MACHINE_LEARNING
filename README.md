# E-COMMERCE_MARKETING_MACHINE_LEARNING
 The goal of marketing in e-commerce is to attract, engage, and retain customers through targeted campaigns, improving customer lifetime value ,CLV and brand loyalty..
# Customer Segmentation & Churn Prediction  

## Overview  
This repository contains two machine learning projects focused on customer analytics:  
1. **Customer Segmentation using RFM Analysis & K-Means Clustering**  
2. **Customer Churn Prediction in an E-commerce Dataset**  

Both projects involve data preprocessing, feature engineering, model training, and evaluation to extract meaningful insights and improve business decision-making.  

---  

## Project 1: Customer Segmentation (UK-Based Online Retail Dataset)  

### Objective  
Customer segmentation is crucial in modern customer-centric marketing. This project uses **RFM analysis (Recency, Frequency, Monetary value)** and **K-Means Clustering** to segment customers based on purchasing behavior.  

### Dataset  
- The dataset is from an online UK-based retail store, covering transactions from **01/12/2009 to 09/12/2010**.  
- The company mainly sells unique, all-occasion giftware.  
- Many customers are wholesalers.  

### Methodology  
1. **Data Preprocessing**:  
   - Handled missing values and outliers.  
   - Filtered relevant transactions.  
2. **Feature Engineering (RFM Analysis)**:  
   - **Recency**: How recently a customer made a purchase.  
   - **Frequency**: How often a customer purchases.  
   - **Monetary Value**: Total spending of a customer.  
3. **Standardization**:  
   - Scaled the RFM values using **MinMaxScaling**.  
4. **Clustering using K-Means**:  
   - Chose **3 clusters** for segmentation.  
   - Identified customer groups based on spending patterns.  
5. **Prediction for Incoming Customers**:  
   - Trained a model to predict the segment of new customers.  
6. **Modeling Tools Used**:  
   - **TensorFlow** for training the predictive model.  
   - **Scikit-Learn** for clustering.  

---

## Project 2: Customer Churn Prediction (E-commerce Dataset)  

### Objective  
This project aims to predict whether a customer will churn using an **e-commerce dataset** with over **5,000 records** and **20 features**.  

### Methodology  
1. **Data Cleaning & Preprocessing**:  
   - Handled missing values, duplicate entries, and outliers.  
   - Standardized features for consistency.  
2. **Feature Selection**:  
   - Analyzed relationships between **churn** and different attributes.  
   - Kept optimal features to improve accuracy.  
3. **Model Training & Evaluation**:  
   - **Trained multiple models** to determine the best one:  
     - Logistic Regression  
     - Decision Trees  
     - Random Forests  
     - Support Vector Machine (SVM)  
   - **Best Model**: Random Forest with **94% accuracy**.  
   - **Evaluation Metrics**:  
     - Confusion Matrix  
     - Precision, Recall, F1-score  

---

## Results & Insights  

- **Customer Segmentation:**  
  - Successfully categorized customers into 3 segments based on purchasing behavior.  
  - Helps businesses target different groups with personalized marketing strategies.  

- **Churn Prediction:**  
  - Achieved **94% accuracy** in predicting customer churn using **Random Forest**.  
  - Provides insights into the key factors contributing to customer churn.  

---
## Technologies Used
- Python

- Pandas, NumPy, Matplotlib, Seaborn (Data Analysis & Visualization)

- Scikit-Learn (Machine Learning Models)

- TensorFlow (Predictive Model for Customer Segmentation)
