from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib

# Important for Flask/web applications
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import joblib
import io
import base64

app = Flask(__name__)

# Load trained ML model
model = joblib.load("house_price_model.pkl")

# Load historical data
df = pd.read_csv("house_data.csv", sep=",")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    chart = None
    size = None
    bedrooms = None
    age = None

    if request.method == "POST":

        # --------------------------------
        # 1. GET USER INPUT
        # --------------------------------

        size = float(request.form["size"])
        bedrooms = int(request.form["bedrooms"])
        age = float(request.form["age"])

        # --------------------------------
        # 2. NUMPY - SIMPLE DATA ANALYSIS
        # --------------------------------

        average_size = np.mean(df["Size_m2"])
        average_price = np.mean(df["Price_USD"])

        print("Average House Size:", average_size)
        print("Average House Price:", average_price)

        # --------------------------------
        # 3. PANDAS - PREPARE USER DATA
        # --------------------------------

        new_house = pd.DataFrame(
            [[size, bedrooms, age]],
            columns=["Size_m2", "Bedrooms", "Age_years"]
        )

        # --------------------------------
        # 4. SCIKIT-LEARN - PREDICTION
        # --------------------------------

        predicted_price = model.predict(new_house)[0]

        prediction = f"${predicted_price:,.2f}"

        # --------------------------------
        # 5. MATPLOTLIB - VISUALIZATION
        # --------------------------------

        plt.figure(figsize=(8, 5))

        plt.scatter(
            df["Size_m2"],
            df["Price_USD"],
            label="Historical Houses"
        )

        plt.scatter(
            size,
            predicted_price,
            s=180,
            marker="*",
            label="Your Prediction"
        )

        plt.xlabel("House Size (m²)")
        plt.ylabel("House Price (USD)")
        plt.title("House Size vs Price")
        plt.grid(alpha=0.3)
        plt.legend()

        # Save graph into memory instead of a file
        image = io.BytesIO()

        plt.savefig(
            image,
            format="png",
            bbox_inches="tight"
        )

        image.seek(0)

        chart = base64.b64encode(
            image.getvalue()
        ).decode()

        plt.close()

    return render_template(
        "index.html",
        prediction=prediction,
        chart=chart,
        size=size,
        bedrooms=bedrooms,
        age=age
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5050,
        debug=True
    )