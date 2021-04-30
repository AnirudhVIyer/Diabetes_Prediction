# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
from ml_model import Convert

# Load the Random Forest CLassifier model
filename = 'model_diabetes.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = int(request.form['nutrition'])
        nutrition = int(request.form['gender'])   
        mom = int(request.form['mom'])     
        auto = int(request.form['auto'])    
        glu_met = int(request.form['glu_met']) 
        insulin = int(request.form['insulin'])     
        T1 = int(request.form['T1'])
        T2 = int(request.form['T2'])
        hypo = int(request.form['Hypo'])
        pan = int(request.form['pan'])        
        res = int(request.form['res'])   
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        bmi = float(request.form['bmi'])
        h1b = int(request.form['h1b'])





        data = np.array([[age,sex,res,h1b,height,weight,bmi,nutrition,mom,auto,glu_met,insulin,T1,T2,hypo,pan]])
        y = Convert(data)
        
        output = classifier.predict(y)
       

         
        return render_template('result.html', prediction=int(output))




@app.route('/return',methods=['GET'])
def back():
    return render_template('index.html')


if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0', port=9090)