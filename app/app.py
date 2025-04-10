# here i will create the Flask app
# with endpoit for predictions

from flask import Flask, request, jsonify
from utilities import predict, loaded_model

app = Flask(__name__)

# creating /predict endpoint that handles POST requests
@app.post('/predict')

def predict_endpoint():
    try:
        # parse the input data as plain text
        data = request.get_data(as_text=True)
        
        # convert the input string to a list of floats
        features = list(map(float, data.strip('[]').split(',')))
        
        # ensure the input has the correct number of features
        if len(features) != 4:
            return jsonify({'error': 'Invalid input. Expected a vector with 4 elements.'}), 400
        
        # make predictions using the loaded model
        predictions = predict(features, loaded_model)
        
        # return the predictions as a JSON response
        return jsonify({'predicted class': predictions.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000, debug=True)

# can make post requests to this endpoint
# used Postman: POST http://127.0.0.1:4000/predict, raw text [...]