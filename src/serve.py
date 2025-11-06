# serve.py
import argparse
from flask import Flask, request, jsonify
from src.utils import load_model
import pandas as pd




def create_app(model_path):
model = load_model(model_path)
app = Flask(__name__)


@app.route('/ping')
def ping():
return 'pong', 200


@app.route('/predict', methods=['POST'])
def predict():
payload = request.get_json()
# Expect payload to be a single row dict or list of rows
if isinstance(payload, dict):
df = pd.DataFrame([payload])
else:
df = pd.DataFrame(payload)


# NOTE: This example assumes client sends already preprocessed features
probs = model.predict(df)
preds = (probs > 0.5).astype(int)
return jsonify({'predictions': preds.tolist(), 'probabilities': probs.tolist()})


return app




if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--host', default='0.0.0.0')
parser.add_argument('--port', default=5000, type=int)
args = parser.parse_args()
app = create_app(args.model)
app.run(host=args.host, port=args.port)
