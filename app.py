from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model('simple_face_recognition_model.h5')

@app.route('/')
def index():
    return app.send_static_file('index.html')  # Serve the single HTML file

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']  # Fetch the uploaded image
    
    if file:
        # Save the uploaded file
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        
        # Preprocess the uploaded image
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict with the model
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions[0])
        
        # Respond with the result
        response = {'message': f'Predicted class index: {class_index}'}
        return jsonify(response)
    
    return jsonify({'message': 'No file uploaded'}), 400

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)  # Create an 'uploads' folder if it doesn't exist
    app.run(debug=True)
