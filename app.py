from flask import Flask,request,render_template
import joblib
import numpy as np
import sklearn

model = joblib.load('./model/insurance.pkl')
sc_x = joblib.load('./model/scaler_x.pkl')
sc_y = joblib.load('./model/scaler_y.pkl')

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    prediction_result = 0
    edad_request = 0
    
    if request.method == 'POST':
        edad_request = int(request.form['edad'])
        edad_sc = sc_x.transform(np.array([[edad_request]]))
        prediction = model.predict(edad_sc)
        prediction_sc = sc_y.inverse_transform(prediction)
        prediction_result = round(prediction_sc[0][0],2)
        
    return render_template('index.html',edad=edad_request, prediction=prediction_result)

app.run(debug=True)