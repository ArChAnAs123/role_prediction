
import flask
import pickle
from flask import render_template,request
import pandas as pd
import numpy as np
from werkzeug.wrappers import request
# Use pickle to load in the pre-trained model.

import model as m
with open('boost.pkl', 'rb') as f:
    mod = pickle.load(f)

app = flask.Flask(__name__, template_folder='template')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/',methods=['POST'])
def predict():
         input = [float(x) for x in flask.request.form.values()]
         input_variables = [np.array(input)]
         prediction = m.prediction(input_variables)

         return flask.render_template('index.html', result = prediction )    

    

       
       


if __name__ == "__main__":
    app.run( debug= True)    
