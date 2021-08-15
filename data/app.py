# Creating Flask app for model deployment
import numpy as np
from flask import Flask, jsonify, request, render_template
import pickle

app = Flask(__name__)

# Root endpoint
@app.route('/', methods = ['GET', 'POST'])
def home():
    return render_template('index.html')
    
# Predict endpoint
@app.route('/predict', methods = ['POST'])
def predict():
    model = pickle.load(open('model.pickle', 'rb'))

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction_lep = model.predict(final_features)
    
    output = round(prediction_lep[0], 3)
    
    return render_template('index.html', prediction_text='Insurance charge should be $ {}'.format(output))


    
 
if __name__ == "__main__":
    app.run(port = 5000, debug = True)   