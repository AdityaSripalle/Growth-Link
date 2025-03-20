# Growth-Link

---

# Churn Prediction Using Random Forest Classifier

## Objective

This project aims to predict customer churn (i.e., whether a customer will exit or stay with the company) using a dataset containing various customer attributes. The solution uses a machine learning model (Random Forest Classifier) to analyze customer data and predict the likelihood of churn.

### Key steps:
1. Load and preprocess the dataset.
2. Handle missing values.
3. Encode categorical features.
4. Split the data into training and testing sets.
5. Standardize the data.
6. Train a Random Forest Classifier model.
7. Evaluate the model's performance.
8. Display feature importance.

---

## Steps to Run the Project

### Prerequisites

Ensure you have the following installed:
- Python 3.x
- Required libraries: pandas, numpy, seaborn, matplotlib, scikit-learn

### 1. Install dependencies
If you haven’t already installed the required libraries, you can do so by running the following command:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

### 2. Dataset
Download the dataset (`Churn_Modelling.csv`) and place it in the same directory as the Python script.

### 3. Running the Script
To run the project, simply execute the Python script that contains the following code:

```bash
python churn_prediction.py
```

This will perform the following tasks:
1. Load the dataset.
2. Clean and preprocess the data.
3. Train a Random Forest Classifier model.
4. Evaluate the model's performance with accuracy score, classification report, and confusion matrix.
5. Display the feature importance.

---

## Code Explanation

1. Data Loading:  
   We load the dataset using `pandas.read_csv()` to bring in the customer data.

2. Data Preprocessing:  
   - Unnecessary columns like `RowNumber`, `CustomerId`, and `Surname` are dropped from the dataset.
   - Missing numerical values are imputed using the median strategy.
   - Categorical columns like `Geography` and `Gender` are label encoded for the model.

3. Feature Selection:  
   We separate the features (`X`) and target (`y`), with `Exited` being the target variable.

4. Data Splitting:  
   The dataset is split into training and testing sets using `train_test_split()` with 80% for training and 20% for testing.

5. Feature Scaling:  
   We standardize the features using `StandardScaler` to normalize the data before feeding it into the model.

6. Model Training:  
   A Random Forest Classifier is trained with 100 estimators on the training set.

7. Model Evaluation:  
   The trained model's performance is evaluated using accuracy, a classification report, and a confusion matrix.

8. Feature Importance:  
   A bar plot is displayed to show the importance of each feature in the model's prediction.

---

## Model Performance Evaluation

After running the model, the following outputs will be displayed:

1. Accuracy: The accuracy score of the model on the test data.
2. Classification Report: Detailed performance metrics such as precision, recall, and F1-score for each class.
3. Confusion Matrix: A confusion matrix showing the number of true positives, true negatives, false positives, and false negatives.

---

## Example Output

```text
Accuracy: 0.85
Classification Report:
               precision    recall  f1-score   support

           0       0.89      0.91      0.90       1600
           1       0.79      0.75      0.77        400

    accuracy                           0.85       2000
   macro avg       0.84      0.83      0.83       2000
weighted avg       0.85      0.85      0.85       2000

Confusion Matrix:
 [[1457   143]
 [ 100   300]]
```

---

## Feature Importance

A bar plot will be displayed showing the relative importance of each feature in predicting customer churn. This helps identify which features are most influential in the model's decision-making process.

---

## License

This project is open source.

---

## Author

Aditya Sripalle

---

## Acknowledgments

- The dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction).
- Thanks to the contributors to the scikit-learn and pandas libraries.


