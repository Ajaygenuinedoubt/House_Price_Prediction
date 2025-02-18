# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
@st.cache_data
def load_data():
    # Replace with the path to your dataset
    df = pd.read_csv("train.csv")
    return df

# Preprocessing the dataset
@st.cache_data
def preprocess_data(df):
    # Select important features (you can select based on importance analysis)
    features = ["MSSubClass", "MSZoning", "LotArea", "Neighborhood", "BldgType", 
                "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "TotalBsmtSF", 
                "GrLivArea", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", 
                "GarageCars", "GarageArea", "SaleCondition"]
    
    # Label encoding for categorical variables
    label_encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    X = df[features]
    y = df["SalePrice"]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, X, y, label_encoders

# Train the model
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Predict the house price
def predict_price(model, user_input, label_encoders):
    # Convert categorical inputs into the same format as training data
    for feature, encoder in label_encoders.items():
        if feature in user_input:
            user_input[feature] = encoder.transform([user_input[feature]])[0]

    prediction = model.predict(pd.DataFrame([user_input]))[0]
    return prediction

# Main function to build the dashboard
def main():
    st.title("Advanced House Price Prediction Dashboard -> [Ajay Kumar Jha] ")
    
    df = load_data()
    X_train, X_test, y_train, y_test, X, y, label_encoders = preprocess_data(df)
    
    model = train_model(X_train, y_train)
    
    st.header("Enter the House Features for Price Prediction:")
    
    # Create input widgets for important features
    MSSubClass = st.selectbox("MSSubClass", sorted(df["MSSubClass"].unique()))
    MSZoning = st.selectbox("MSZoning", sorted(df["MSZoning"].unique()))
    LotArea = st.number_input("LotArea", min_value=int(df["LotArea"].min()), max_value=int(df["LotArea"].max()))
    Neighborhood = st.selectbox("Neighborhood", sorted(df["Neighborhood"].unique()))
    BldgType = st.selectbox("BldgType", sorted(df["BldgType"].unique()))
    OverallQual = st.slider("OverallQual", 1, 10)
    OverallCond = st.slider("OverallCond", 1, 10)
    YearBuilt = st.slider("YearBuilt", int(df["YearBuilt"].min()), int(df["YearBuilt"].max()))
    YearRemodAdd = st.slider("YearRemodAdd", int(df["YearRemodAdd"].min()), int(df["YearRemodAdd"].max()))
    TotalBsmtSF = st.number_input("TotalBsmtSF", min_value=0, max_value=int(df["TotalBsmtSF"].max()))
    GrLivArea = st.number_input("GrLivArea", min_value=0, max_value=int(df["GrLivArea"].max()))
    FullBath = st.selectbox("FullBath", sorted(df["FullBath"].unique()))
    HalfBath = st.selectbox("HalfBath", sorted(df["HalfBath"].unique()))
    BedroomAbvGr = st.selectbox("BedroomAbvGr", sorted(df["BedroomAbvGr"].unique()))
    KitchenAbvGr = st.selectbox("KitchenAbvGr", sorted(df["KitchenAbvGr"].unique()))
    GarageCars = st.selectbox("GarageCars", sorted(df["GarageCars"].unique()))
    GarageArea = st.number_input("GarageArea", min_value=0, max_value=int(df["GarageArea"].max()))
    SaleCondition = st.selectbox("SaleCondition", sorted(df["SaleCondition"].unique()))

    # Dictionary for user inputs
    user_input = {
        "MSSubClass": MSSubClass,
        "MSZoning": MSZoning,
        "LotArea": LotArea,
        "Neighborhood": Neighborhood,
        "BldgType": BldgType,
        "OverallQual": OverallQual,
        "OverallCond": OverallCond,
        "YearBuilt": YearBuilt,
        "YearRemodAdd": YearRemodAdd,
        "TotalBsmtSF": TotalBsmtSF,
        "GrLivArea": GrLivArea,
        "FullBath": FullBath,
        "HalfBath": HalfBath,
        "BedroomAbvGr": BedroomAbvGr,
        "KitchenAbvGr": KitchenAbvGr,
        "GarageCars": GarageCars,
        "GarageArea": GarageArea,
        "SaleCondition": SaleCondition
    }
    
    # Predict and display the result
    if st.button("Predict Price"):
        prediction = predict_price(model, user_input, label_encoders)
        st.success(f"The predicted price for the house is: ${prediction:,.2f}")

# Run the app
if __name__ == "__main__":
    main()
