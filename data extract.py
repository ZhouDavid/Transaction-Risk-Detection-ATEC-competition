import pandas as pd
import os
import random

data_path = './data/full_size'
output_path = './data/small_size'
def extract_data(filename,extract_lines):
	lines = open(data_path+'/'+filename).readlines()
	res = random.sample(lines,extract_lines)
	return res

if __name__ == '__main__':
	sample = extract_data('atec_anti_fraud_train.csv',20000)
	open(output_path+'/sample_atec_anti_fraud_train.csv','w+').writelines(sample)
