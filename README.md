# House_Price_Prediction
House Price Prediction Dashboard using Streamlit
![image](https://github.com/user-attachments/assets/4608177a-4f69-4638-9d73-2962a341bbc3)

## Advanced House Price Prediction Dashboard

This project is a Streamlit-based interactive web application designed to predict house prices based on various features of a house. It leverages machine learning to build a predictive model and provides an intuitive user interface for users to input house features and get price predictions.

## Features

Interactive user interface for inputting house features.

Real-time house price prediction based on user inputs.

Uses a trained RandomForestRegressor model for accurate predictions.

Automatic data preprocessing, including label encoding for categorical variables.

Efficient handling of data through caching to optimize performance.

## Table of Contents

Installation

Dataset

Project Structure

Model Training

Feature Engineering

## Prerequisites
Ensure you have the following installed on your local machine:

Python 3.7+
pip

# Clone the repository:


git clone    https://github.com/Ajaygenuinedoubt/House_Price_Prediction
Navigate to the project directory:



cd house-price-prediction-dashboard

## Install the required dependencies:


pip install -r requirements.txt
Run the application:


streamlit run houseprediction.py

## Dataset

The dataset used in this project is a house price dataset, which contains various features like the number of rooms, lot area, year built, and more. The target column is SalePrice, which represents the price of the house.

For this example, the dataset is named train.csv. You can replace it with your own dataset if you have one. Ensure the dataset is placed in the same directory as the code, or update the file path in the load_data function accordingly.

## Project Structure


├── app.py              # Main application file

├── train.csv           # House price dataset

├── requirements.txt    # Required dependencies

├── README.md           # Project README

## Explanation of Key Files:

app.py: The main Streamlit app, which handles the data loading, preprocessing, model training, and UI components.

train.csv: The dataset used for training the model and making predictions. You can use any other dataset by modifying the load_data function in the app.py file.

requirements.txt: Contains the necessary dependencies required to run the project.

## Usage

Once the project is installed, follow these steps to use the application:

Run the Streamlit app:


streamlit run app.py

You will be redirected to your default browser where the app will be hosted locally.

## The app interface allows you to input various features of a house, including:

MSSubClass (building class)

MSZoning (zoning classification)

LotArea (size of the lot)

Neighborhood, BldgType, and other relevant features.

After filling in the details, click on the "Predict Price" button to get the predicted house price.

## Model Training

The model used in this project is a RandomForestRegressor, a robust machine learning algorithm for regression tasks. It was chosen for its ability to handle complex datasets with both categorical and numerical variables.

## Model Training Steps:

The dataset is first loaded and preprocessed. Categorical variables are label-encoded using LabelEncoder.

The data is split into training and testing sets using train_test_split (80% training, 20% testing).

The RandomForestRegressor is trained on the training data.

The trained model is cached to improve the app's performance and speed up predictions.

## Feature Engineering

The following features were selected for training the model based on their importance in predicting house prices:

MSSubClass, MSZoning, LotArea, Neighborhood, BldgType

OverallQual, OverallCond, YearBuilt, YearRemodAdd

TotalBsmtSF, GrLivArea, FullBath, HalfBath

BedroomAbvGr, KitchenAbvGr, GarageCars, GarageArea

SaleCondition

These features cover various aspects of the house, including its size, condition, location, and amenities.

## How to Use the App

Enter the house features into the user interface:
Select or input values for each feature such as lot area, year built, the number of bedrooms, and so on.
Click on the "Predict Price" button.
The predicted price will be displayed on the screen based on the model's output.
Contributing
If you'd like to contribute to this project:

## Fork the repository.

Create a new branch:

git checkout -b my-feature-branch

Make your changes.
Commit your changes:

git commit -m "Add new feature"
Push to the branch:

git push origin my-feature-branch
Open a pull request on GitHub.
