import UAV as uav
import pandas as pd
import os
import json
from collections import Counter
import numpy as np
from sklearn.preprocessing import StandardScaler
import math
from collections import Counter
import pickle
import data_processing as dp
import utils



mapped_subcat = {0:"manual mode (0)",
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




def get_indexable_meta(meta_json):
	indexable_meta = {}

	for m in meta_json:
		temp_id = m["id"]
		m.pop("id")
		indexable_meta[temp_id] = m
		
	return indexable_meta

with open(json_file, 'r') as inputFile:
	meta_json = json.load(inputFile)
indexable_meta = get_indexable_meta(meta_json)
	

def get_filtered_ids():

	ulogs_downloaded = os.listdir(ulog_folder)
	# ulogs_downloaded_hex = os.listdir(ulog_folder_hex)


	# ulogs_downloaded = ulogs_downloaded_hex + ulogs_downloaded 


	drone_ids = [u[:-4] for u in ulogs_downloaded]

	filtered_ids = [u for u in drone_ids if indexable_meta[u]["duration"] != "0:00:00"]
	filtered_ids = [u for u in drone_ids if indexable_meta[u]["type"] == "Quadrotor"]

	distribution = [indexable_meta[u]["type"] for u in filtered_ids]
	print(Counter(distribution))
			
	return filtered_ids

def split_data(df, feat_name):
	current_index = 0
	prev_value = df["mode"].iloc[0]
	prev_index = 0
	log_X = []
	log_y = [prev_value]

	for i, row in df.iterrows():
		next_value = int(row["mode"])
		if next_value != prev_value:
			log_X.append((prev_value, df.iloc[prev_index:i].reset_index()[["timestamp", feat_name]]))
			# log_X.append((prev_value, feat_name))
			prev_value = next_value
			prev_index = i
			log_y.append(next_value)

	log_X.append((prev_value, df.iloc[prev_index:len(df)].reset_index()[["timestamp", feat_name]]))
	# log_X.append((prev_value, feat_name))

	return log_X, log_y

def extract_dfs(mUAV, table_name, feat_name):
	data = mUAV.get_desired_feats(table_name, feat_name)
	modes = mUAV.modes_nearest_indices(table_name)
	timestamps = mUAV.get_time_stamp(table_name)

	tmp_dict = []

	for i, mode in enumerate(modes):
		tmp_dict.append({"timestamp": timestamps[i], "mode": mode, feat_name: data[i]})

	df = pd.DataFrame.from_dict(tmp_dict)

	return df

def generate_data(feats):
	# ids = get_filtered_ids()
	with open("../../../../UAV_ML/full_parsed_7_multi.txt", "rb") as f:
		full_parsed_split = dp.split_features(pickle.load(f))
		temp_ids = list(full_parsed_split.keys())

	ids = [u for u in temp_ids if indexable_meta[u]["type"] == "Quadrotor"]
	print(len(ids))

	X = []
	y = []
	mapped_X = {}
	mapped_y = {}
	success_count = 0

	path = ulog_folder

	for i, id in enumerate(ids):
		print(f"Log {i}/{len(ids)}")
		print(f"Success count: {success_count}")
		print(id)

		mUAV = uav.UAV(id, path)
		mapped_X[id] = {}

		modes = []
		data = {}

		counter = 0
		for feat in feats:
			strings = feat.split(" ")
			table_name = strings[0]
			feat_name = strings[2]
			try:
				df = extract_dfs(mUAV, table_name, feat_name)
				counter += 1
			except:
				print(feat_name)
				break

		if counter == len(feats):
			for feat in feats:
				strings = feat.split(" ")
				table_name = strings[0]
				feat_name = strings[2]

				df = extract_dfs(mUAV, table_name, feat_name)
				log_X, log_y = split_data(df, feat_name)

				data[feat] = log_X
				modes.append(log_y)


			
			lens = [len(m) for m in modes]
			min_ind = np.argmin(np.array(lens))
			min_list = modes[min_ind]

			mapped_X[id] = [{} for i in min_list]

			for key, value in data.items():
				temp_index = 0
				# mapped_X[id][key] = {}

				for i, x in enumerate(value):		
					if x[0] == min_list[temp_index]:
						mapped_X[id][temp_index][key] = x[1]
						temp_index += 1	

					if temp_index == len(min_list):
						break		

			mapped_y[id] = min_list

			success_count += 1


	return mapped_X, mapped_y


def get_equal_distribution(X, y, min_instances=1000):
	counts = dict(Counter(y))
	# print(counts)
	new_X, new_y = {}, []
	modes = [key for key, value in counts.items() if value > min_instances - 1]
	counter = {m:0 for m in modes}
	keys = list(X.keys())

	i = 0

	while modes:
		current_val = y[i]
		
		if current_val in modes:
			counter[current_val] += 1
			new_X[keys[i]] = X[keys[i]]
			new_y.append(current_val)

			if counter[current_val] == min_instances:
				modes.remove(current_val)
				del counter[current_val]

		i += 1

	# print(Counter(new_y))

	# print(new_X.keys())

	# print(new_X[list(new_X.keys())[0]])

	return new_X, new_y


def apply_dur_threshold(X, y):
	print(len(X))
	total_mins = []
	total_maxes = []
	keys = list(X.keys())

	for x in X.values():
		indiv_mins = []
		indiv_maxes = []
		for key, value in x.items():
			indiv_mins.append(value["timestamp"].min())
			indiv_maxes.append(value["timestamp"].max())
		total_mins.append(min(indiv_mins))
		total_maxes.append(max(indiv_maxes))


	durations = list(np.array(total_maxes) - np.array(total_mins))
	durations = list(map(lambda x : utils.micros_to_secs(x), durations))

	threshold_ind = [i for i, d in enumerate(durations) if d >= 10]
	threshold_keys = list(np.array(keys)[threshold_ind])


	new_X = {}
	for key in threshold_keys:

		new_X[key] = X[key]

	new_y = list(np.array(y)[threshold_ind])

	print(len(new_X))
	return new_X, new_y

def to_category(y):
	new_y = [mapped_label[mapped_cat[mapped_subcat[i]]] for i in y]

	return new_y


def remap_y(y):
	counts = dict(Counter(y))
	new_mapping = {}
	new_y = []
	for i, value in enumerate(counts.keys()):
		new_mapping[value] = i

	for i in y:
		new_y.append(new_mapping[i])

	new_mapping = {value:key for key,value in new_mapping.items()}

	return new_y, new_mapping


def extending_lists(mapped_X, mapped_y, feats):
    X = {}
    y = {}
    y_list = []

    for key, value in mapped_X.items():
        counter = 0
        for i, sample in enumerate(value):
            if len(list(sample.keys())) == len(feats):
                X[f"{key} | {counter}"] = sample
                y[f"{key} | {counter}"] = mapped_y[key][i]
                y_list.append(mapped_y[key][i])
                counter += 1


    return X, y, y_list
	

def preprocess_data(mapped_X, mapped_y, feats, equal_dist=False, chunking=False):
	X, y, y_list = extending_lists(mapped_X, mapped_y, feats)

	X, y_list = apply_dur_threshold(X, y_list)

	if equal_dist:
		X, y_list = get_equal_distribution(X, y_list)

	if chunking:
		X, ids_intervals, y_list = dp.get_x_min_chunks(X, y)

	# X = {k: X[k] for k in list(X)[:100]}

	new_X = dp.timestamp_bin(X)
	# new_X = dp.timestamp_bin_local(X)

	print(np.array(new_X).shape)

	new_y, new_mapping = remap_y(y_list)

	print(np.array(new_y).shape)

	return new_X, new_y, new_mapping


def standardize_data(X_train, X_test):
	scaler = StandardScaler()

	num_instances = X_train.shape[0]
	num_features = X_train.shape[1]
	num_instances_test = X_test.shape[0]
	num_features_test = X_test.shape[1]

	if len(X_train.shape) == 3:
		num_times = X_train.shape[2]
		num_times_test = X_test.shape[2]
		scaler = scaler.fit(X_train.reshape(num_instances * num_times, num_features))
		X_train = scaler.transform(X_train.reshape(num_instances * num_times, num_features))
		X_test = scaler.transform(X_test.reshape(num_instances_test * num_times_test, num_features_test))

		X_train = X_train.reshape(num_instances, num_features, num_times)
		X_test = X_test.reshape(num_instances_test, num_features_test, num_times_test)
	else:
		scaler = scaler.fit(X_train)
		X_train = scaler.transform(X_train)
		X_test = scaler.transform(X_test)


	return X_train, X_test

def main():
	# feats = ["vehicle_local_position | x", "vehicle_local_position | y",
	#          "vehicle_local_position | z", "vehicle_attitude_setpoint | roll_body",
	#          "vehicle_attitude_setpoint | pitch_body", "vehicle_attitude_setpoint | yaw_body",
	#          "manual_control_setpoint | z", "vehicle_gps_position | alt", "battery_status | temperature"]

	feats = ["vehicle_local_position | x", "vehicle_local_position | y",
	         "vehicle_local_position | z"]

	# Generate mapped data
	# mapped_X, mapped_y = generate_data(feats)

	# with open("new_mapped_X_xyz.txt", "wb") as f:
	# 	pickle.dump(mapped_X, f)

	# with open("new_mapped_y_xyz.txt", "wb") as f:
	# 	pickle.dump(mapped_y, f)



	# Preprocessing mapped data
	with open("new_mapped_X_xyz.txt", "rb") as f:
		mapped_X = pickle.load(f)

	with open("new_mapped_y_xyz.txt", "rb") as f:
		mapped_y = pickle.load(f)

	new_X, new_y, new_mapping = preprocess_data(mapped_X, mapped_y, feats)

	with open("X_data_xyz.txt", "wb") as f:
		pickle.dump(new_X, f)

	with open("y_data_xyz.txt", "wb") as f:
		pickle.dump(new_y, f)









	# sample_id = "b6bf05ef-15b7-4eb5-b94b-96a0f95c3730"
	# mUAV = uav.UAV(sample_id, ulog_folder)

	# map_temp = {sample_id:{}}
	# map_temp_y = {sample_id:{}}

	# feat = feats[0]
	# strings = feat.split(" ")
	# table_name = strings[0]
	# feat_name = strings[2]
	# df_1 = extract_dfs(mUAV, table_name, feat_name)
	# log_X, log_y = split_data(df_1, feat_name)

	# map_temp[sample_id][feat] = log_X

	# print(len(map_temp[sample_id][feat]))

	# feat = feats[5]
	# strings = feat.split(" ")
	# table_name = strings[0]
	# feat_name = strings[2]
	# df_2 = extract_dfs(mUAV, table_name, feat_name)
	# log_X, log_y = split_data(df_2, feat_name)

	# map_temp[sample_id][feat] = log_X
	# map_temp_y[sample_id][feat] = log_y


	# print(len(map_temp[sample_id][feat]))
	# print(map_temp_y[sample_id][feat])

if __name__ == "__main__":
	main()




