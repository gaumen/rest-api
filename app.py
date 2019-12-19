
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import joblib
import traceback
from flask_restful import reqparse
app = Flask(__name__)

@app.route("/", methods=['GET'])
def hello():
    return "hey"

@app.route('/predict', methods=['POST'])
#@app.route('/', methods=['GET'])
def predict():
	if lr:
		try:
			json = request.get_json()
			'''parser = reqparse.RequestParser()
			parser.add_argument('query')
			args = parser.parse_args()
			user_query = args['query']   ''' 
			#return request.form 
			'''query = pd.get_dummies(pd.DataFrame(json))
			query = query.reindex(columns=model_columns, fill_value=0)'''
			temp=list(json[0].values())
			vals=np.array(temp)
			print(vals)
			prediction = lr.predict(temp)
			print("here:",prediction)        
			return jsonify({'prediction': str(prediction[0])})

		except:        
			return jsonify({'trace': traceback.format_exc()})
	else:
		print ('Train the model first')
		return ('No model here to use')
    


if __name__ == '__main__':
    lr = joblib.load("model.pkl") 
    model_columns = joblib.load("model_cols.pkl")
    
    app.run(debug=True)
    
