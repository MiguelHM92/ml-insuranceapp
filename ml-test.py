import joblib
import numpy as np
import sklearn

model = joblib.load('./model/insurance.pkl')
sc_x = joblib.load('./model/scaler_x.pkl')
sc_y = joblib.load('./model/scaler_y.pkl')

edad = int(input('Ingrese la edad del paciente: '))
edad_sc = sc_x.transform(np.array([[edad]]))

prediction = model.predict(edad_sc)

prediction_sc = sc_y.inverse_transform(prediction)
print(f'Los gastos médicos para un paciente con {edad} años resulta: $ {prediction_sc[0][0]:.2f}')