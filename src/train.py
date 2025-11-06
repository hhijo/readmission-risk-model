# train.py
import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import lightgbm as lgb
from src.utils import save_model, train_val_test_split
from src.preprocess import basic_preprocess




def main(data_path, output_dir, target_col="readmit_30"):
output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(data_path)
train_df, val_df, test_df = train_val_test_split(df, target_col)


X_train, y_train = basic_preprocess(train_df, target_col)
X_val, y_val = basic_preprocess(val_df, target_col)


train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)


params = {
'objective': 'binary',
'metric': 'auc',
'verbosity': -1,
'boosting_type': 'gbdt',
'seed': 42,
'learning_rate': 0.05,
'num_leaves': 31
}


model = lgb.train(params, train_data, valid_sets=[val_data], early_stopping_rounds=50, num_boost_round=1000)


save_model(model, output_dir / 'model.joblib')


# sample evaluation on validation
preds = model.predict(X_val)
pred_labels = (preds > 0.5).astype(int)
print('Precision:', precision_score(y_val, pred_labels))
print('Recall:', recall_score(y_val, pred_labels))
print('AUC:', roc_auc_score(y_val, preds))




if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--data-path', required=True, help='CSV file with raw data (single file for example)')
parser.add_argument('--output-dir', default='models/')
args = parser.parse_args()
main(args.data_path, args.output_dir)
