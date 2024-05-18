import argparse
import pandas as pd
from utils import read_pickle
from sklearn.multioutput import RegressorChain
import pickle
from catboost import Pool, CatBoostRegressor
import numpy as np
import os


target_mapping = {
	'NbWords': 'W',
	'NbChars': 'C',
	'LevSim': 'L',
	'WordRank': 'WR',
	'DepTreeDepth': 'DTD',
	'ReplaceLevSim': 'RL',
}

def bucketize(value, bucket_size=0.05):
    '''Round value to bucket_size to reduce the number of different values'''
    return round(round(value / bucket_size) * bucket_size, 10)

def compute_metrics(eval_pred, target):
	predictions, labels = eval_pred
	colwise_rmse = np.sqrt(np.mean((labels - predictions) ** 2, axis=0))

	res = {}
	for i in range(labels.shape[1]):
		res[f"{target[i]}_CORR"] = np.corrcoef(labels[:, i], predictions[:, i])[0][1]
		res[f"{target[i]}_RMSE"] = colwise_rmse[i]
	res["MCRMSE"] = np.mean(colwise_rmse)
	return res


def train(args):

	CONFIG = read_pickle(args.config_file)

	features = CONFIG["features"]
	target = CONFIG["target"]
	train = pd.read_csv(args.train_file)
	train.dropna(inplace=True)
	test = pd.read_csv(args.eval_file)
	test.fillna(0, inplace=True)

	dtrain = Pool(train[features], label=train[target])
	dvalid = Pool(test[features], label=test[target])

	params = {'learning_rate': 0.1, 'depth': 6, 
		        'loss_function': 'RMSE',  'eval_metric': 'RMSE'}

	if CONFIG["model"] == "multi-out-regressor":
		params['eval_metric'] = 'MultiRMSE'
		params['loss_function'] = 'MultiRMSE'
		multioutputregressor = CatBoostRegressor(**params)
		multioutputregressor.fit(dtrain, eval_set=dvalid, use_best_model=True, verbose=0)
	elif CONFIG["model"] == "single-out-regressor":
		multioutputregressor = CatBoostRegressor(**params)
		multioutputregressor.fit(dtrain, eval_set=dvalid, use_best_model=True, verbose=0)
	else:
		multioutputregressor = RegressorChain(base_estimator=CatBoostRegressor(**params), order=CONFIG["order"]).fit(train[features], train[target])
	
	out_multi = multioutputregressor.predict(test[features])
	if len(out_multi.shape) == 1:
		out_multi = np.expand_dims(out_multi, axis=1)
	results = compute_metrics((test[target].to_numpy(), out_multi), target)
	print(results)

	with open(f'{os.path.abspath(os.path.join(args.config_file, os.pardir))}/model.pkl', "wb") as f:
		pickle.dump(multioutputregressor, f)


def eval(args):

	CONFIG = read_pickle(args.config_file)

	with open(f'{os.path.abspath(os.path.join(args.config_file, os.pardir))}/model.pkl', "rb") as f:
		regressor = pickle.load(f)

	if len(CONFIG["target"]) > 1:
		eval_mode = "multi-output"
	else:
		eval_mode = "single"

	test = pd.read_csv(args.eval_file)
	test.fillna(0, inplace=True)

	if args.use_ari:
		print("Here")
		test["SG"] = test["SG_ARI"]
		suffix = "_ari"
	else:
		suffix = ""

	target =  CONFIG["target"]
	predictions = regressor.predict(test[ CONFIG["features"]])

	if len(predictions.shape) == 1:
		predictions = np.expand_dims(predictions, axis=1)

	results = compute_metrics((test[target].to_numpy(), predictions), target)
	print(results)

	if eval_mode == "multi-output":
		with open(f'{os.path.abspath(os.path.join(args.config_file, os.pardir))}/output{suffix}.txt', "w") as f:
			for index, row in test.iterrows():
				tokens = (" ").join([f"{target_mapping[target[i]]}_{predictions[index, i]:.2f}" for i in range(len(target))])
				f.write(f"{tokens} {row['Complex']}\n")

	else:
		with open(f'{os.path.abspath(os.path.join(args.config_file, os.pardir))}/output_single{suffix}.txt', "w") as f:
			for index, row in test.iterrows():
				f.write(f"{target_mapping[CONFIG['target'][0]]}_{predictions[index, 0]:.2f} {row['Complex']}\n")


if __name__ == "__main__":
	parser = argparse.ArgumentParser("Arguments for Training/Evaluating XGBModel")
	parser.add_argument("--train_file", help="path to training data (a dataframe)", required=True)
	parser.add_argument("--eval_file", help="path to evaluation data (a dataframe)", required=True)
	parser.add_argument("--config_file", help="Config dictionary", required=True)
	parser.add_argument("--mode", help="train or eval", default="train-and-eval")
	parser.add_argument('--use-ari', action='store_true')
	
	args = parser.parse_args()
	
	if args.mode == "train":
		train(args)
	elif args.mode == "train-and-eval":
		train(args)
		eval(args)
	else:
		eval(args)