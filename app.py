from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('random_forest_classsifier.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template('air_quality.html')

@app.route('/predict',methods = ['POST', 'GET'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    pred = model.predict(final_features)
    if pred == 0:
        prediction = '_'
    elif pred == 1:
        prediction = 'Good'
    else:
        prediction = 'Moderate'

    output = prediction
    return render_template('air_quality.html', pred='The air quality is {}'.format(output))
if __name__ == '__main__':
    app.run(debug=True)

