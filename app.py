from flask import Flask, request, Response, send_from_directory
from flask_cors import CORS, cross_origin
import os
import json
app = Flask(__name__, static_folder="build")
cors = CORS(app, origins=["*", "http://localhost:3000/"])

from IrisClassification import TrainModel, PredictionModel

@app.route('/', methods=['GET', 'POST', 'OPTIONS'], defaults={'path': ''})
@app.route('/<path:path>')
@cross_origin(origins='*')
def serve(path):
    TrainModel()
    if path != "" and os.path.exists(app.static_folder + '/' + path): # type: ignore
        return send_from_directory(app.static_folder, path) # type: ignore
    else:
        return send_from_directory(app.static_folder, 'index.html') # type: ignore

@app.route('/prediction/', methods=['POST', 'OPTIONS'])
@cross_origin(origins='*')
def prediction():
    if request.method == 'POST':
        specs = request.json["Data"] # type: ignore
        spL = float(specs['SepalLength'])
        spW = float(specs['SepalWidth'])
        ptL = float(specs['PetalLength'])
        ptW = float(specs['PetalWidth'])

        iris = PredictionModel([spL, spW, ptL, ptW])
        

        return Response(json.dumps({"result": iris.tolist() }), status=200, mimetype='application/json')
    return Response('{"result": "OK"}', status=200, mimetype='application/json')

if __name__ == '__main__':
    app.run(use_reloader=True, port=5000, threaded=True)