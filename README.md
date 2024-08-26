# Breast Cancer Diagnostic Tool

## Project Overview
To create a web application to host a ML model that can accurately predict breast cancer diagnosis based on breast mass sample data.  
![application interface] (https://github.com/IramHasan/machine-learning-projects/blob/main/cancer-tissue-predictor/assets/breast-cancer-predictor.png)

## Data Collection
The data is taken from the Diagnostic Wisconsin Breast Cancer database.
Features are from digitzed images of fine needle aspirate (FNA) of breast tissue mass, and attempt to describe the characteristics of cell nuclei present in the image. There are 30 features present in the raw data used to create a classification model.

## Data Cleaning
Prior to model training the data was cleaned by removing the "Unnamed: 32" column in the raw data which contained 0 non-null values. We also removed the id feature from the data as it will not be relevant for analysis.
Furthermore, we performed a transformation of the diagnosis values mapping begnign (B) to 0 and malignant (M) to 1. 

## Logistic Regression Model
![logistic regression equation](https://github.com/user-attachments/assets/4588de16-3f27-4200-9aed-5c670754f05e)  

We will take numerical inputs in the form of the 30 features which are traditionally used to determine cell characteristics. This will be the input for a logistic regression model. The logistic regression model will provide a categorical value either begnign or malignant. 
The above equation is the logisitc regression equation used to transform our regression data to values between 0 and 1.

## Model Creation
### Predictors vs Target
The data was first designated into predictors and target values, with the diagnosis column being the target and the remaining features being the predictors.

### Scaling Model data
From the raw data, theres a wide variety of scales of different magnitudes. To make the model more intuitive and easier to understand / train the data was normalized.

### Train/Test Split
The data was split 80/20 to training and test sets with randomized shuffling (random_state=42).

## Model Evaluation
|              | PRECISION | RECALL | F1-SCORE | SUPPORT | 
|--------------|-----------|--------|----------|---------|
| 0            | 0.97      | 0.99   | 0.98     | 71      |
| 1            | 0.98      | 0.95   | 0.96     | 43      |
| accuracy     |           |        | 0.97     | 114     |
| macro avg    | 0.97      | 0.97   | 0.97     | 114     |
| weighted avg | 0.97      | 0.97   | 0.97     | 114     |

The model's F1 and accuracy scores are 0.97 which suggests that the model is highly accurate when it comes to the categorization of malignant or begnin cells. The classes of the analaysis are not class balanced based on the support scores so the macro average F1 score is a good predictor of accuracy. 

## Considerations
There could have been exploratory analysis of the data and raw data. Addition of visualizations to the project document. This could help illustrate characteristics of the raw and cleaned data set such as the null values, the class balancing, etc.

A better approach to developing the model would be to first do training and test splits before scaling the model. We would have performed 2 fit_transform operations on both the training and test predictors separately. This method would better prevent data leakage.
