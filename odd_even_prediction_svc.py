from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the previously saved model
model = load_model('oddEvenSum.h5')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from request
    data = request.get_json(force=True)
    try:
        # Extract the two integers from the request data
        num1 = int(data['num1'])
        num2 = int(data['num2'])

        # Validate the input integers
        if not (0 <= num1 <= 255 and 0 <= num2 <= 255):
            return jsonify({'error': 'Both numbers must be between 0 and 255.'}), 400

        # Convert integers to 8-bit binary arrays
        binary1 = [int(x) for x in f'{num1:08b}']
        binary2 = [int(x) for x in f'{num2:08b}']

        # Concatenate binary arrays
        binary_input = binary1 + binary2

        # Reshape for model prediction
        x_input = np.array(binary_input).reshape(1, -1)  # Reshape for single prediction

        # Make prediction with the model
        prediction = model.predict(x_input)
        is_true = (prediction > 0.5).astype(int).item() == 1

        # Calculate the sum of the two numbers
        total_sum = num1 + num2

        # Return result and sum
        return jsonify({
            'prediction': is_true,
            'sum': total_sum
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
