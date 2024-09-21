# Diabetes Disease Prediction

This project is focused on predicting whether a patient is diabetic or not using a dataset containing several health metrics. The project involves preprocessing data, building a predictive model using machine learning, and providing a web-based interface for user interaction. The web application allows users to input specific data points and predict the likelihood of diabetes.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model](#model)
- [Files in the Repository](#files-in-the-repository)
- [How to Run](#how-to-run)
- [Technologies Used](#technologies-used)
- [Results and Metrics](#results-and-metrics)

## Project Overview

The main objective of this project is to build a machine learning model that can predict whether a person is diabetic or non-diabetic based on several health-related attributes. We utilize logistic regression for classification and implement a grid search to fine-tune the hyperparameters.

The model is deployed using Flask to provide a user-friendly web interface where users can input health data and get a prediction.

## Dataset

The dataset used in this project is located in the `Dataset/diabetes.csv` file. The dataset includes the following features:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI (Body Mass Index)
- Diabetes Pedigree Function
- Age
- Outcome (Target variable: 1 for Diabetic, 0 for Non-Diabetic)

Some of the columns, such as `Glucose`, `Insulin`, `Blood Pressure`, etc., had values of 0, which were replaced with their respective mean values since 0 is not a valid value for those features.

## Model

The model used for prediction is a **Logistic Regression** model. The steps involved in training the model include:

- **Data Preprocessing**: Handling missing or incorrect values (e.g., 0 values for certain features).
- **Feature Scaling**: Standard scaling is applied to standardize features before feeding them into the model.
- **Model Training**: The model was trained using a logistic regression classifier with hyperparameter tuning via grid search.
- **Performance Metrics**: Accuracy, Precision, Recall, and F1-Score were calculated to evaluate the model.

## Files in the Repository

- `Dataset/diabetes.csv`: The dataset used for training and testing the model.
- `Model/modelForPrediction.pkl`: The trained machine learning model saved using pickle.
- `Model/standardScaler.pkl`: The scaler used to standardize the input data.
- `Notebook/Diabetes_Disease_Prediction.ipynb`: The Jupyter Notebook containing the code for data preprocessing, training, and evaluation of the model.
- `application.py`: The Flask application to serve the model and handle user inputs for predictions.
- `templates/`: Folder containing HTML files for the front-end of the web application.
  - `home.html`: Home page of the web application.
  - `index.html`: Main page for interacting with the model.
  - `single_prediction.html`: Page displaying the prediction result.

## How to Run

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/ali-samin/Diabetes_Disease_Prediction.git
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the Flask application:
   ```bash
   python application.py
4. Open a web browser and navigate to http://127.0.0.1:5000/ to access the web interface.

## Technologies Used

- Python: Core programming language.
- Flask: For developing the web application.
- Scikit-learn: For machine learning algorithms.
- Pandas: For data manipulation.
- NumPy: For numerical computations.
- Matplotlib and Seaborn: For data visualization.

## Results and Metrics
The model was evaluated using several performance metrics:

- Accuracy: Measures the percentage of correctly predicted cases.
- Precision: The ratio of correctly predicted positive observations to the total predicted positives.
- Recall: The ratio of correctly predicted positive observations to all observations in the actual class.
- F1-Score: A weighted average of Precision and Recall.

Confusion Matrix:

Predicted: Non-Diabetic	Predicted: Diabetic
Actual: Non-Diabetic	TP	FP
Actual: Diabetic	FN	TN
