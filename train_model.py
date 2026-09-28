import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib

# --------------------------------
# 1. LOAD THE HOUSE DATA
# --------------------------------

df = pd.read_csv("house_data.csv", sep=",")
print("COLUMN NAMES:")
print(df.columns.tolist())

print("\nHOUSE DATA")
print(df)

# --------------------------------
# 2. PREPARE INPUTS AND OUTPUT
# --------------------------------

X = df[["Size_m2", "Bedrooms", "Age_years"]]
y = df["Price_USD"]

# --------------------------------
# 3. TRAIN MACHINE LEARNING MODEL
# --------------------------------

model = LinearRegression()

model.fit(X, y)

# --------------------------------
# 4. CHECK MODEL PERFORMANCE
# --------------------------------

predictions = model.predict(X)

score = r2_score(y, predictions)

print("\nMODEL PERFORMANCE")
print(f"R² Score: {score:.3f}")

# --------------------------------
# 5. TEST WITH A NEW HOUSE
# --------------------------------

new_house = pd.DataFrame(
    [[150, 3, 3]],
    columns=["Size_m2", "Bedrooms", "Age_years"]
)

predicted_price = model.predict(new_house)[0]

print("\nTEST PREDICTION")
print("House Size: 150 m²")
print("Bedrooms: 3")
print("Age: 3 years")
print(f"Predicted Price: ${predicted_price:,.2f}")

# --------------------------------
# 6. SAVE TRAINED AI MODEL
# --------------------------------

joblib.dump(model, "house_price_model.pkl")

print("\nModel saved successfully!")