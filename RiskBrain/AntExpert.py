import numpy as np
import pandas as pd
import Sampler
import Configure
import Preprocessor

class AntExpert:
	def __init__(self,model_type):
		self.model_type = model_type

	def read_train_test_data(self,train_path,test_path):
		"""
		read csv data as pandas dataframe
		:return: dataframe
		"""
		try:
			self.train_data = pd.read_csv(train_path,index_col = 0)
			self.test_data = pd.read_csv(test_path,index_col=0)
		except FileNotFoundError:
			print('file not found')

	def get_filldable_features(self,missing_thr):
		"""get no missing feature names"""
		self.no_missing_features=[]
		i = 1
		while i < len(self.test_data.columns):
			name = self.test_data.columns[i]
			if self.train_data[name].isnull().sum() == 0:
				self.no_missing_features.append(name)
			i += 1

		"""get small missing feature names"""
		i = 1
		while i < len(testb_data.columns):
			name = testb_data.columns[i]
			train_missing_rate = train_data[name].isnull().sum() / train_data.shape[0]
			test_missing_rate = testb_data[name].isnull().sum() / testb_data.shape[0]
			if 0 < train_missing_rate <  and abs(test_missing_rate - train_missing_rate) < 0.1:
				small_missing_features.append(name)
			i += 1
		return no_missing_features+no_missing_features

	def get_feature_importance(self,important_thr):
		feature_score_files = ['../result/xgb_feature_scores.csv', '../result/lgb_feature_scores2.csv']
		"""get important features"""
		for file in feature_score_files:
			features = set(pd.read_csv(file, index_col=0, header=None).sort_values(by=1, ascending=False).iloc[:important_thr,
						   0].index.tolist())
			all_important_features = all_important_features | features
			if common_important_features:
				common_important_features = common_important_features & features
			else:
				common_important_features = features
		return all_important_features,common_important_features

	def preprocess(self,handle_unlabel = True,fillnan=True,missing_thr = 0.3,fill_type = 'mean',
				   standarlize=False,feature_type=0,important_thr=100,including_date=False,date_converted=False):
		"""
		:param handle_unlabel: if True,convert all -1 label to 1
		:param fillnan: if True,fill all small missing features with fill_type method
		:param missing_thr: threshold to determine which feature is filldable
		:param standarlize: if True,using standarlization
		:param feature_type: 0 represents using all 297 provided features,1 means using all important features+filldable_features,
		2 means just using filldable features, 3 means just use no_missing features
		:param important_thr: top important_thr scored features are important 
		:param including_date: if True, will add date as feature
		:param date_converted: if True, will converted date to the day of a month
		:return: None
		"""
		if handle_unlabel:
			self.train_data.label = self.train_data.label.apply(lambda x:1 if x==-1 else x)
		if fillnan:
			filldable_features = self.get_filldable_features(missing_thr)
			if fill_type=='mean':
				fill_values = self.train_data[filldable_features].mean()
				self.train_data[filldable_features] = self.train_data[filldable_features].fillna(fill_values)
			else:
				print('fill method not supported')
		if standarlize:
			from sklearn import preprocessing
			self.scaler = preprocessing.StandardScaler()
			self.train_data = self.scaler.fit_transform(self.train_data)
			self.test_data = self.scaler.transform(self.test_data)
		self.train_x = self.train_data.drop(columns=['date','label'])
		self.train_y = self.train_data.label
		
		if including_date:
			self.train_x.date = self.train_data.date



		self.all_imporant_features = self.get_feature_importance(important_thr) # get common_important and all_important features in both xgb and lgb



	def get_tpr_from_fpr(fpr_array, tpr_array, target):
		import bisect
		fpr_index = np.where(fpr_array == target)
		assert target <= 0.01, 'the value of fpr in the custom metric function need lt 0.01'
		if len(fpr_index[0]) > 0:
			return np.mean(tpr_array[fpr_index])
		else:
			tmp_index = bisect.bisect(fpr_array, target)
			fpr_tmp_1 = fpr_array[tmp_index - 1]
			fpr_tmp_2 = fpr_array[tmp_index]
			if (target - fpr_tmp_1) > (fpr_tmp_2 - target):
				tpr_index = tmp_index
			else:
				tpr_index = tmp_index - 1
			return tpr_array[tpr_index]

	def eval_metric(labels, pred):
		fpr, tpr, _ = metrics.roc_curve(labels, pred, pos_label=1)
		tpr1 = get_tpr_from_fpr(fpr, tpr, 0.001)
		tpr2 = get_tpr_from_fpr(fpr, tpr, 0.005)
		tpr3 = get_tpr_from_fpr(fpr, tpr, 0.01)
		return 0.4 * tpr1 + 0.3 * tpr2 + 0.3 * tpr3





if __name__ == '__main__':
	config = Configure()
	expert = AntExpert(model_type='lightgbm')
	expert.read_train_test_data('../data/full_size/atec_anti_fraud_train.csv',
								'../data/full_size/atec_anti_fraud_test_b.csv')
	expert.sample_train_data(sample_type='oversample')
	expert.preprocess()
	expert.train_test_split(percent = 0.2)
	expert.cv_train(cv=5,split_type='random')
	expert.predict()
	expert.write_result_to_file()



