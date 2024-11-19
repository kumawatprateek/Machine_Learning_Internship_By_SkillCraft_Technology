# House Price Prediction

**Task 1** of my **Skillcraft Technology Machine Learning Internship** focuses on predicting house prices using various regression models. The project leverages historical house sale data with features such as **bedrooms**, **bathrooms**, **square footage**, and more to estimate house prices.

## Overview
This project uses a dataset containing real-estate information to predict house prices. Various machine learning models have been implemented to perform this regression task, including:
- **Linear Regression**
- **Random Forest Regressor**
- **Decision Tree Regressor**
- **Gradient Boosting Regressor**

## Project Steps

### 1. Data Cleaning and Preprocessing:
- Selected relevant features including **bedrooms**, **bathrooms**, **square footage**, and more.
- Checked for and removed any missing values.
- Scaled the feature values for improved performance in regression models.

### 2. Model Building:
Implemented the following regression models:
- **Linear Regression**
- **Random Forest Regressor**
- **Decision Tree Regressor**
- **Gradient Boosting Regressor**

### 3. Model Evaluation:
The models were evaluated using:
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **Accuracy** (for classification-like comparison)

### 4. New User Input for Prediction:
The project allows users to input house details manually and receive price predictions based on the trained models.

## Results
Here are the accuracy scores for the models:
- **Random Forest**: 99%
- **Decision Tree**: 98%
- **Linear Regression**: 99%
- **Gradient Boosting Regressor**: 99%

The **MAE** and **RMSE** for each model were also calculated to compare their performance.

## How to Use
1. Clone this repository to your local machine.
2. Open the `House_Price_Prediction.ipynb` file in Jupyter Notebook or Google Colab.
3. Run the cells to train the models and test them.
4. Input new data in the provided section to predict house prices.

## Dataset
The dataset used in this project contains features such as:
- **price**, **bedrooms**, **bathrooms**, **sqft_living**, **sqft_lot**, **floors**, **waterfront**, **view**, **condition**

You can replace the dataset with your own real-estate data for further customization.

## Requirements
Install the following libraries to run the project:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

You can install them using pip:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

