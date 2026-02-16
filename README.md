# road-accident-survey-data-analytics
A comprehensive Data Analytics project focused on identifying road accident patterns and predicting accident severity in India using Machine Learning and OLAP operations.


**Problem Statement:** Road accidents are increasing annually, leading to significant loss of life and property. This project addresses the lack of proper prediction models and user-friendly systems to help users understand and take quick action regarding accident risks.

**Objective:** To collect and analyze road accident data to identify patterns and trends, implement machine learning models for accident prediction, and develop an intuitive system for accessing actionable insights.

**Data Source:** The project utilizes the India Road Accident Dataset from Kaggle, consisting of 59,999 records and 66 attributes.

**Key Modules:**

    ◦ Module I (Preprocessing): Includes data acquisition, removing duplicates, handling missing values, and normalization/standardization of attributes like Number_of_Vehicles.
    
    ◦ Module II (Modeling): Implementation of Association Rule Mining (Apriori, FP-Growth), Classification (CART, ID3, C4.5), and Clustering (K-Means, DBSCAN).
    
    ◦ Module III (Testing): Evaluation of models based on Accuracy, Precision, Recall, and F1 Score, with the CART model achieving a high accuracy of 0.8667.
    
    ◦ Module IV (Deployment): A user-friendly UI designed for predicting accident severity (Fatal, Serious, or Slight) and visualizing statistical trends through line charts and 3D visualizations
