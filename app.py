from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load trained model
model = keras.models.load_model("plant_disease_model.h5")

# Define class labels
class_labels = ["Apple Rust", "Potato Blight", "Healthy", "Other"]  # Update based on your dataset

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image = Image.open(io.BytesIO(file.read()))
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return jsonify({"class": predicted_class, "confidence": confidence})

if __name__ == '__main__':
    app.run(debug=True)
