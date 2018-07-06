class Configure:
	def __init__(self):


class LGBMConfigure:
	def __init__(self):
		self.model_name = 'lightgbm'
		self.model_params = {'n_estimators': 100, 'num_leavs': 350, 'max_depth': 9, 'learning_rate': 0.1}
		self.using_default = False

class RandomForestConfigure():
	def __init__(self):
		self.using_default = False

class GBDTConfigure():
	def __init__(self):
		self.using_default = False