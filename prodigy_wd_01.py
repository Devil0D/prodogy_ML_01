import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

try:
    train_data = pd.read_csv('./dataset/train.csv')
    test_data = pd.read_csv('./dataset/test.csv')
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# Separate features and target
X_train_full = train_data.drop(['SalePrice', 'Id'], axis=1)
y_train = train_data['SalePrice']
test_ids = test_data['Id']
X_test_full = test_data.drop(['Id'], axis=1)

# Combine train and test for consistent preprocessing
combined = pd.concat([X_train_full, X_test_full], axis=0)

# Identify numerical and categorical columns
categorical_cols = combined.select_dtypes(include=['object']).columns.tolist()
numerical_cols = combined.select_dtypes(include=['int64', 'float64']).columns.tolist()

if 'SalePrice' in numerical_cols:
    numerical_cols.remove('SalePrice')
if 'Id' in numerical_cols:
    numerical_cols.remove('Id')

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[ 
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# --- Training and Validation split ---
X_train, X_valid, y_train_small, y_valid = train_test_split(X_train_full, y_train, test_size=0.2, random_state=42)

# Fit the pipeline on training subset
model.fit(X_train, y_train_small)

# Validate the model
y_pred_valid = model.predict(X_valid)
mse = mean_squared_error(y_valid, y_pred_valid)
rmse = np.sqrt(mse)
r2 = r2_score(y_valid, y_pred_valid)

print(f"Validation MSE: {mse:.2f}")
print(f"Validation RMSE: {rmse:.2f}")
print(f"Validation R² Score: {r2:.2f}")

# --- Retrain model on full training data ---
model.fit(X_train_full, y_train)

# Predict on test data
test_preds = model.predict(X_test_full)

# Create submission file
submission = pd.DataFrame({'Id': test_ids, 'SalePrice': test_preds})
submission.to_csv('submission.csv', index=False)

print("🎯 Submission file 'submission.csv' created successfully!")

# --- Prediction based on Custom Inputs ---
def predict_house_price(sqft, bedrooms, bathrooms):
    # Create a dummy row based on the first row of the training data
    dummy_row = X_train_full.iloc[0:1].copy()
    
    # Set your custom values for sqft, bedrooms, and bathrooms
    dummy_row['GrLivArea'] = sqft      # Set the square footage
    dummy_row['BedroomAbvGr'] = bedrooms  # Set the number of bedrooms
    dummy_row['FullBath'] = bathrooms    # Set the number of full bathrooms

    # Predict the house price using the trained model
    predicted_price = model.predict(dummy_row)

    # Return the predicted price
    return predicted_price[0]

# Example of using the function to predict house price
sqft = 2500  # Example square footage
bedrooms = 4  # Example number of bedrooms
bathrooms = 2  # Example number of full bathrooms

predicted_price = predict_house_price(sqft, bedrooms, bathrooms)
print(f"Predicted House Price for {sqft} sqft, {bedrooms} bedrooms, {bathrooms} bathrooms: ${predicted_price:,.2f}")
