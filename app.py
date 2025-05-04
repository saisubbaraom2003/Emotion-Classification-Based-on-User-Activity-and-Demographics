from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Define the categories used during training
gender_categories = ['Female', 'Male', 'Non-binary']
platform_categories = ['Instagram', 'Twitter', 'Facebook', 'LinkedIn', 'Whatsapp', 'Telegram', 'Snapchat']

# Initialize encoders/scalers with fixed fitting
scaler = MinMaxScaler()
encoder = OneHotEncoder(categories=[gender_categories, platform_categories], drop='first', sparse_output=False, handle_unknown='ignore')

# Fit dummy data to mimic original training
dummy_num_data = pd.DataFrame([[10, 10, 1], [100, 1000, 50]], columns=["Age", "Daily_Usage_Time (minutes)", "Posts_Per_Day"])
scaler.fit(dummy_num_data)

dummy_cat_data = pd.DataFrame([
    ['Female', 'Instagram'],
    ['Male', 'Twitter']
], columns=["Gender", "Platform"])
encoder.fit(dummy_cat_data)

# Emotion label mapping
emotion_labels = {
    0: "Happiness",
    1: "Neutral",
    2: "Boredom",
    3: "Anxiety",
    4: "Anger"
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Get form inputs
            age = float(request.form["age"])
            usage_time = float(request.form["usage_time"])
            posts = float(request.form["posts"])
            gender = request.form["gender"]
            platform = request.form["platform"]

            # Numeric preprocessing
            num_input = pd.DataFrame([[age, usage_time, posts]], columns=["Age", "Daily_Usage_Time (minutes)", "Posts_Per_Day"])
            scaled_num = scaler.transform(num_input)

            # Categorical preprocessing
            cat_input = pd.DataFrame([[gender, platform]], columns=["Gender", "Platform"])
            encoded_cat = encoder.transform(cat_input)

            # Final input
            final_input = np.concatenate((scaled_num, encoded_cat), axis=1)

            # Make prediction
            prediction = model.predict(final_input)[0]
            predicted_emotion = emotion_labels[prediction]

            return render_template("index.html", prediction=predicted_emotion)

        except Exception as e:
            return render_template("index.html", prediction=f"Error: {str(e)}")

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
