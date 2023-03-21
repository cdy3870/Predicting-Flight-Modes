import os
import pandas as pd
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import math
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, adjusted_mutual_info_score, roc_auc_score, f1_score
import pickle
from collections import Counter
import argparse
from sklearn.model_selection import KFold, StratifiedKFold
import random 
from itertools import combinations
import csv
import random
import numpy as np
import flight_mode_preprocess as fmp

# Classical models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Seed
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



feats = ["vehicle_local_position | x", "vehicle_local_position | y",
		 "vehicle_local_position | z", "vehicle_attitude_setpoint | roll_body",
		 "vehicle_attitude_setpoint | pitch_body", "vehicle_attitude_setpoint | yaw_body",
		 "manual_control_setpoint | z", "vehicle_gps_position | alt", "battery_status | temperature"]

mapped_subcat = {-1: "other",
				0:"manual mode (0)",
				1:"altitude control mode (1)",
				2:"position control mode (2)",
				3:"mission mode (3)",
				4:"loiter mode (4)",
				5:"return to launch mode (5)",
				6:"RC recovery (6)",
				8:"free slot (8)",
				9:"free slot (9)",
				10:"acro mode (10)",
				11:"free slot (11)",
				12:"descend mode (no position control) (12)",
				13:"termination mode (13)",
				14:"offboard (14)",
				15:"stabilized mode (15)",
				16:"rattitude (16)",
				17:"takeoff (17)",
				18:"land (18)",
				19:"follow (19)",
				20:"precision land with landing target (20)",
				21:"orbit in circle (21)",
				22:"takeoff, transition, establish loiter (22)"}

mapped_cat = {"RC recovery (6)":"auto",
				"manual mode (0)":"manual",
				"altitude control mode (1)":"guided",
				"position control mode (2)":"guided",
				"mission mode (3)":"auto",
				"loiter mode (4)":"auto",
				"return to launch mode (5)":"auto",
				"free slot (8)":"undefined",
				"free slot (9)":"undefined",
				"acro mode (10)":"guided",
				"free slot (11)":"undefined",
				"descend mode (no position control) (12)":"auto",
				"termination mode (13)":"auto",
				"offboard (14)": "auto",
				"stabilized mode (15)":"guided",
				"rattitude (16)":"guided",
				"takeoff (17)":"auto",
				"land (18)":"auto",
				"follow (19)":"auto",
				"precision land with landing target (20)":"auto",
				"orbit in circle (21)":"guided",
					"takeoff, transition, establish loiter (22)":"auto"}

mapped_label = {"guided":0, "auto":1, "manual":2}                

ulog_folder = "../../../../../../work/uav-ml/px4-Ulog-Parsers/dataDownloaded"
ulog_folder_hex = "../../../../../../work/uav-ml/px4-Ulog-Parsers/dataDownloadedHex"
json_file = "../../../../../../work/uav-ml/px4-Ulog-Parsers/MetaLogs.json"

##### PYTORCH DATA FUNCTIONS #####

class UAVDataset(Dataset):
	"""
	UAV Dataset class

	...

	Attributes
	----------
	X : list
		Parsed and feature engineered time series data
	y : list
		UAV labels

	Methods
	-------
	len():
		gets the size of the dataset
	getitem(index):
		gets an indexed instance
	"""
	def __init__(self, X, y):
		self.X = X
		self.y = y
						
	def __len__(self):
		return len(self.X)
	
	def __getitem__(self, index):
		return torch.Tensor(self.X[index]), torch.tensor(self.y[index]).float()   


def get_dataloaders(data):
	# Form datasets and dataloaders
	train_dataset = UAVDataset(data["X_train"], data["y_train"])
	test_dataset = UAVDataset(data["X_test"], data["y_test"])

	train_loader = DataLoader(train_dataset,
							  batch_size=8,
							  shuffle=False)
	test_loader = DataLoader(test_dataset,
							  batch_size=1,
							  shuffle=False)

	return train_loader, test_loader


##### CONCAT/CLASSICAL MODELS ##### 

def reshape_data(X_train, X_test):
	X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
	X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

	return X_train, X_test

def apply_concat_models(classes, data, model_name="SVM", verbose=True, reshape=True):
	if reshape:
		X_train, X_test = reshape_data(data["X_train"], data["X_test"])
	else:
		X_train, X_test = data["X_train"], data["X_test"]

	if model_name == "SVM":
		model = SVC()
	elif model_name == "RFC":
		model = RandomForestClassifier()
	elif model_name == "XGBC":
		model = XGBClassifier()

	model = model.fit(X_train, data["y_train"])
	y_pred = model.predict(X_test)		
	counts = Counter(y_pred)
	report = classification_report(data["y_test"], y_pred, target_names=classes, output_dict=True)
	macro_f1 = f1_score(data["y_test"], y_pred, average='macro')
	conf_mat = confusion_matrix(data["y_test"], y_pred)

	if verbose:
		print(counts)
		print(report)
		print(macro_f1)
		print(conf_mat)

	if model_name in ["RFC", "XGBC"]:
		return counts, macro_f1, counts, report, conf_mat, model.feature_importances_, y_pred


	return counts, macro_f1, counts, report, conf_mat, [], y_pred


##### LSTM ##### 

class LSTM(nn.Module):
	"""
	LSTM class

	...

	Attributes
	----------
	input_size : int
		number of instances
	hidden_size : int
		hidden size of LSTM layer
	num_classes : int
		number of classes to predict
	num_layers : int
		number of LSTM layers

	Methods
	-------
	forward():
		forward propagation of LSTM
	"""
	def __init__(self, input_size, hidden_size, num_classes, num_layers):
		super(LSTM, self).__init__()
		self.LSTM = nn.LSTM(input_size=input_size,
							hidden_size=hidden_size,
							num_layers=num_layers,
							batch_first=True)
		self.fc = nn.Linear(hidden_size, num_classes)
	
	def forward(self, x):
		# print(x[0])
		_, (hidden, _) = self.LSTM(x)
		out = hidden[-1]
		# print(out)
		# print(out.shape)
		out = self.fc(out)
		# print(out)
		# print(out.shape)
		return out

def get_model(input_size, hidden_size = 128, num_classes=2, num_layers=1):
	hidden_size = 128
	num_layers = 1
	model = LSTM(input_size=input_size,
				 hidden_size=hidden_size,
				 num_classes=num_classes,
				num_layers=num_layers)

	return model


def apply_NNs(classes, model, train_loader, test_loader, params, verbose=True, test_index=None):
	'''
	Trains the LSTM 

			Parameters:
					model (object): Pytorch model
					train_loader (object): Iterable train loader
					test_loader (object): Iterable test loader
					params (dict): Model and training params

			Returns:
					pred_from_last_epoch (list) : Predictions from the last epoch
	'''
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = model.to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
	progress_bar = tqdm(range(params["num_epochs"]))
	
	num_correct = 0
	true_size = 0

	pred_from_last_epoch = []
	
	for epoch in progress_bar:
		y_true = []
		y_pred = []
		for phase in ("train", "eval"):
			if phase == "train":
				model.train()
				data_loader = train_loader
			else:
				model.eval()
				data_loader = test_loader
				
			for (_, data) in enumerate(data_loader):
				optimizer.zero_grad()
				inputs = data[0].to(device)
				targets = data[1]
				# print(targets)
				targets = targets.to(device)

				with torch.set_grad_enabled(phase=="train"):
					predictions = model(inputs)

					# print(predictions.shape)
					loss = criterion(predictions, targets.long())

					if phase == "train":
						loss.backward()
						# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
						optimizer.step()

				out, max_indices = torch.max(predictions, dim=1)

				if phase == "eval":
					y_true += targets.tolist()
					y_pred += max_indices.tolist()

					# print(predictions)
				
				# num_correct += torch.sum(max_indices == targets.long()).item()

				# true_size += targets.size()[0]
				# print(targets.size())

		if phase == "eval" and (epoch + 1) % params["num_epochs"]/4 == 0 :
			counts = Counter(y_pred)
			report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
			auc_score = 0
			# auc_score = roc_auc_score(y_true, y_pred, multi_class='ovo')
			macro_f1 = f1_score(y_true, y_pred, average='macro')
			conf_mat = confusion_matrix(y_true, y_pred)

			if verbose:
				print(counts)
				print(classification_report(y_true, y_pred, target_names=classes))
				print(auc_score)
				print(conf_mat)


		if phase == "eval" and epoch == params["num_epochs"] - 1:
			pred_from_last_epoch = y_pred

	return pred_from_last_epoch, auc_score, macro_f1, counts, model, report, conf_mat



# def generate_true(y_true, y_pred, cat_subcat_test):
# 	new_true = [cat_subcat_test[i][1] for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)) if y_p == y_t]


def train_two_part(classes, model, train_loader, test_loader, params, cat_subcat_test, verbose=True, test_index=None):
	'''
	Trains the LSTM 

			Parameters:
					model (object): Pytorch model
					train_loader (object): Iterable train loader
					test_loader (object): Iterable test loader
					params (dict): Model and training params

			Returns:
					pred_from_last_epoch (list) : Predictions from the last epoch
	'''

	for (_, data) in enumerate(train_loader):
		print(data[0].tolist())

	# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# model = model.to(device)
	# criterion = nn.CrossEntropyLoss()
	# optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
	# progress_bar = tqdm(range(params["num_epochs"]))
	
	# num_correct = 0
	# true_size = 0

	# pred_from_last_epoch = []
	
	# model.train()
	# for epoch in progress_bar:			
	# 	for (_, data) in enumerate(train_loader):
	# 		optimizer.zero_grad()
	# 		inputs = data[0].to(device)
	# 		targets = data[1]
	# 		# print(targets)
	# 		targets = targets.to(device)

	# 		with torch.set_grad_enabled(phase=="train"):
	# 			predictions = model(inputs)

	# 			# print(predictions.shape)
	# 			loss = criterion(predictions, targets.long())
	# 			loss.backward()
	# 			optimizer.step()


				
	# for (_, data) in enumerate(test_loader):
	# 	inputs = data[0].to(device)
	# 	targets = data[1]
	# 	out, max_indices = torch.max(predictions, dim=1)
	# 	y_pred += max_indices.tolist()

	return pred_from_last_epoch, auc_score, macro_f1, counts, model, report, conf_mat






def modify_for_folds(X, y, mapping, classes, n_folds=5):
	new_X = []
	new_y = []
	removed_classes = set()

	counts = Counter(y)
	modified_counts = {key:value for key, value in dict(counts).items() if value >= n_folds}
	viable_keys = list(modified_counts.keys())

	for i, y in enumerate(y):
		if y in viable_keys:
			new_X.append(X[i])
			new_y.append(y)
		else:
			removed_classes.add(y)

	print(removed_classes)

	new_classes = [mapped_subcat[mapping[i]] for i in range(len(mapping)) if i not in removed_classes]
	print(new_classes)

	return new_X, new_y, new_classes

def create_other_class(X, y, mapping, classes, n_samples=500):
	new_X = []
	new_y = []
	removed_classes = set()

	counts = Counter(y)

	print(counts)
	modified_counts = {key:value for key, value in dict(counts).items() if value >= n_samples}
	viable_keys = list(modified_counts.keys())

	for i, y in enumerate(y):
		if y not in viable_keys:
			new_y.append(len(viable_keys))
		else:
			new_y.append(y)

		new_X.append(X[i])


	# print(removed_classes)

	new_counts = list(Counter(new_y))

	print(new_counts)

	new_classes = [mapped_subcat[mapping[i]] for i in range(len(new_counts) - 1) if i not in removed_classes]
	new_classes.append("other")
	print(new_classes)

	return new_X, new_y, new_classes


def append_y_to_x(X_train, X_test, y_train, y_test):
	X_train, X_test = reshape_data(X_train, X_test)
	X_train = X_train.tolist()
	X_test = X_test.tolist()

	for x, y in zip(X_train, y_train):
		x.append(y)

	for x, y in zip(X_test, y_test):
		x.append(y)

	# print(X_train[0])

	return np.array(X_train), np.array(X_test)

def main():   
	# X, y = fmp.preprocess_data()

	# with open("X_data.txt", "rb") as f:
	# 	asdf = pickle.load(f)

	# print(len(asdf))




	parser = argparse.ArgumentParser()
	parser.add_argument("-k", "--n_folds", type=int, help="number of folds in k-fold cross validation", default=5)
	parser.add_argument("-l", "--learning_rate", type=float, help="learning rate", default=0.001)
	parser.add_argument("-e", "--n_epochs", type=int, help="number of epochs (divisible by 4)", default=100)
	parser.add_argument('-csv','--csv', type=str, help='csv file name output')
	parser.add_argument("-d", "--description", type=str, help="description of experiment", required=True)
	parser.add_argument("-t", "--target", type=str, help="subcategory or category", required=True)
	parser.add_argument("-m", "--model", type=str, help="model (LSTM, SVM, RFC, XGBC)", default="LSTM")

	parser.add_argument("-eq", "--equal", action='store_true', help="equal distribution")
	parser.add_argument("-ch", "--chunked", action='store_true', help="chunked data")
	parser.add_argument("-ap", "--append", action='store_true', help="append category or subcategory", default=False)
	
	parser.add_argument("-tw", "--twopart", action='store_true', help="if two part model is used", default=False)
	# parser.add_argument("-tw-type", "--twotype", type=str, help="two part step (category, subcategory)", default=None)


	args = parser.parse_args()


	if args.equal:
		with open("X_data_equal.txt", "rb") as f:
			X = pickle.load(f)

		with open("y_data_equal.txt", "rb") as f:
			y_subcat = pickle.load(f)

		with open("mapping_equal.txt", "rb") as f:
			mapping = pickle.load(f)

	else:
		if args.chunked:
			with open("X_data_chunked.txt", "rb") as f:
				X = pickle.load(f)

			with open("y_data_chunked.txt", "rb") as f:
				y_subcat = pickle.load(f)
		else:
			with open("X_data.txt", "rb") as f:
				X = pickle.load(f)
			with open("y_data.txt", "rb") as f:
				y_subcat = pickle.load(f)



		with open("mapping.txt", "rb") as f:
			mapping = pickle.load(f)

	print(np.array(X).shape)
	print("------------------------------------ Parameters ------------------------------------")
	print("Description: " + args.description)


	if args.target == "category":
		# Convert subcategory back to its original id, then use that to get category
		y_cat = fmp.to_category([mapping[i] for i in y_subcat])
		cat_subcat = [(y_c, y_s) for y_c, y_s in zip(y_cat, y_subcat)]
		y = y_cat

		# Convert category to its labels
		reversed_mapping = {value:key for key, value in mapped_label.items()}
		classes = [reversed_mapping[i] for i in list(Counter(y_cat).keys())]

	elif args.target == "subcategory":
		y = y_subcat
		classes = [mapped_subcat[mapping[i]] for i in range(len(mapping))]
		X, y, classes = create_other_class(X, y, mapping, classes)

	kf = StratifiedKFold(n_splits=args.n_folds)
	splits = kf.split(X, y)
	fold = 0
	num_classes = len(Counter(y))
	

	report_stack = np.zeros((args.n_folds, 3, len(classes)))
	# auc_stack = []
	macro_f1_stack = []
	total_conf_mat = np.zeros((len(classes), len(classes)))
	predictions_mapping = {}
	reshape = True


	if args.csv:
		file_name = args.csv
		with open(file_name, 'a', newline='') as csvfile:
			csv_writer = csv.writer(csvfile)
			csv_writer.writerow(["\n"])            
			csv_writer.writerow([f"Description: {args.description}, {args.model}"])

	for train_index, test_index in splits:
		print("------------------------------------ " + "Fold: " + str(fold) + " ------------------------------------")

		X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
		y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

		# Appending
		if args.append:
			reshape = False
			if args.target == "subcategory":
				y_cat_train = fmp.to_category([mapping[i] for i in y_train])
				y_cat_test = fmp.to_category([mapping[i] for i in y_test])
				X_train, X_test = append_y_to_x(X_train, X_test, y_cat_train, y_cat_test)
			elif args.target == "category":
				y_subcat_train = np.array(y_subcat)[train_index]
				y_subcat_test = np.array(y_subcat)[test_index]
				X_train, X_test = append_y_to_x(X_train, X_test, y_subcat_train, y_subcat_test)


		# TWO PART MODEL TEST
		if args.twopart:
			# counts, macro_f1, counts, report, conf_mat, feature_import = apply_concat_two_part(classes, cat_data, subcat_data, model=args.model)
						
			if args.target == "category":
				X_train, X_test = fmp.standardize_data(X_train, X_test)
				cat_data = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}				
				counts, macro_f1, counts, report, conf_mat, _, y_pred = apply_concat_models(classes, cat_data, model_name=args.model)
				predictions_mapping.update(dict(zip(test_index, y_pred)))
				with open("predictions_mapping.txt", "wb") as f:
					pickle.dump(predictions_mapping, f)

			elif args.target == "subcategory":
				with open("predictions_mapping.txt", "rb") as f:
					predictions_mapping = pickle.load(f)

				X_train, X_test = fmp.standardize_data(X_train, X_test)
				y_pred_test = [predictions_mapping[ind] for ind in test_index]
				y_pred_train = [predictions_mapping[ind] for ind in train_index]					
				X_train, X_test = append_y_to_x(X_train, X_test, y_pred_train, y_pred_test)
				
				print(X_train.shape)
				print(y_train.shape)
				subcat_data = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
				preds = apply_concat_models(classes, subcat_data, model_name=args.model, reshape=False)
		else:
			X_train, X_test = fmp.standardize_data(X_train, X_test)
			data = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}	

			if args.model in ["LSTM", "CNN"]:
				train_loader, test_loader = get_dataloaders(data)
				input_size = X_train.shape[2]
				model = get_model(input_size, num_classes=num_classes)
				params = {"lr": args.learning_rate, "num_epochs":args.n_epochs}
				_, auc_score, macro_f1, counts, model, report, conf_mat = apply_NNs(classes, model, train_loader, test_loader, params)
				# _, auc_score, macro_f1, counts, model, report, conf_mat = train_two_part(classes, model, train_loader, test_loader, params, cat_subcat[test_index])

			else:
				counts, macro_f1, counts, report, conf_mat, feat_import, _ = apply_concat_models(classes, data, model_name=args.model, reshape=reshape)
				sorted_import = feat_import.argsort()
				features_sorted = np.array([feat for feat in feats for i in range(50)])[sorted_import]
				avg_rankings = {}
				for i, feat in enumerate(features_sorted.tolist()):
					if feat in avg_rankings:
						avg_rankings[feat] += i
					else:
						avg_rankings[feat] = i

				avg_rankings = {key:value/50 for key, value in avg_rankings.items()}
				sorted_avg_rankings = dict(sorted(avg_rankings.items(), key=lambda x : x[1]))
				sorted_avg_rankings_ls = [(key, value) for key, value in sorted_avg_rankings.items()]






		if args.csv:
			file_name = args.csv
			with open(file_name, 'a', newline='') as csvfile:
				csv_writer = csv.writer(csvfile)
				csv_writer.writerow(["\n"])            
				csv_writer.writerow(["Fold : " + str(fold)])
				csv_writer.writerow([""] + [c for c in classes])
				csv_writer.writerow(["Precision"] + [round(report[c]["precision"], 4) for c in classes])
				csv_writer.writerow(["Recall"] + [round(report[c]["recall"], 4) for c in classes])
				csv_writer.writerow(["F-score"] + [round(report[c]["f1-score"], 4) for c in classes])
				# csv_writer.writerow(["AUROC Score", round(auc_score, 4)])
				csv_writer.writerow(["Macro F1 Score", round(macro_f1, 4)])
				csv_writer.writerow(["True Counts"] + [round(report[c]["support"], 4) for c in classes])
				csv_writer.writerow(["Pred Counts"] + [counts[i] for i, c in enumerate(classes)])
				
				csv_writer.writerow(["\n"])
				csv_writer.writerow(["Feature"] + list(list(zip(*sorted_avg_rankings_ls))[0]))
				csv_writer.writerow(["Average Ranking"] + list(list(zip(*sorted_avg_rankings_ls))[1]))

				csv_writer.writerow(["\n"])
				csv_writer.writerow([""] + [c for c in classes])  
				for i in range(len(conf_mat)):	
					csv_writer.writerow([classes[i]] + conf_mat[i].tolist())

			total_conf_mat = np.add(total_conf_mat, conf_mat)
				

			for i, metric in enumerate(["precision", "recall", "f1-score"]):
				for j, c in enumerate(classes):
					report_stack[fold][i][j] = report[c][metric]

			# auc_stack.append(auc_score)
			macro_f1_stack.append(macro_f1)

		fold += 1


	if args.csv:
		means = np.mean(report_stack, axis=0)
		stds = np.std(report_stack, axis=0)
		macro_mean = np.mean(macro_f1_stack)
		macro_std = np.std(macro_f1_stack)
		conf_mean = total_conf_mat/args.n_folds


		with open(file_name, 'a', newline='') as csvfile:
			csv_writer = csv.writer(csvfile)
			csv_writer.writerow(["\n"]) 
			csv_writer.writerow(["Averages"])           
			csv_writer.writerow([""] + [c for c in classes])
			csv_writer.writerow(["Precision"] + [round(means[0][i], 4) for i in range(num_classes)])
			csv_writer.writerow(["Recall"] + [round(means[1][i], 4) for i in range(num_classes)])
			csv_writer.writerow(["F-score"] + [round(means[2][i], 4) for i in range(num_classes)])
			csv_writer.writerow(["Macro F1 Score", round(macro_mean, 4)])

			csv_writer.writerow(["\n"]) 
			csv_writer.writerow(["Standard Deviations"])           
			csv_writer.writerow([""] + [c for c in classes])
			csv_writer.writerow(["Precision"] + [round(stds[0][i], 4) for i in range(num_classes)])
			csv_writer.writerow(["Recall"] + [round(stds[1][i], 4) for i in range(num_classes)])
			csv_writer.writerow(["F-score"] + [round(stds[2][i], 4) for i in range(num_classes)])
			csv_writer.writerow(["Macro F1 Score", round(macro_std, 4)])

			csv_writer.writerow(["\n"])
			csv_writer.writerow([""] + [c for c in classes])  
			for i in range(len(conf_mat)):	
				csv_writer.writerow([classes[i]] + conf_mean[i].tolist())


if __name__ == "__main__":
	main()
