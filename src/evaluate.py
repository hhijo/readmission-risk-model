# evaluate.py
import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score
from src.utils import load_model
from src.preprocess import basic_preprocess




def main(model_path, test_csv, target_col='readmit_30'):
model = load_model(model_path)
df = pd.read_csv(test_csv)
X_test, y_test = basic_preprocess(df, target_col)


probs = model.predict(X_test)
preds = (probs > 0.5).astype(int)


cm = confusion_matrix(y_test, preds)
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
auc = roc_auc_score(y_test, probs)


print('Confusion matrix:\n', cm)
print(f'Precision: {precision:.3f}')
print(f'Recall: {recall:.3f}')
print(f'AUC: {auc:.3f}')




if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--test', required=True)
args = parser.parse_args()
main(args.model, args.test)
