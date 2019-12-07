import matplotlib.pyplot as plt
import mglearn
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def iris_dataset_plotting():
	iris_dataset_ = load_iris()  # loading iris data set which is pre defined in sklearn kit
	print(iris_dataset_.keys())  # what all information does this data set contains
	X_train, X_test, y_train, y_test = train_test_split(
														iris_dataset_['data'], 
														iris_dataset_['target'], 
														random_state=0
														)

	iris_dataframe = pd.DataFrame(
									X_train, 
									columns=iris_dataset_['feature_names']
								) # getting dataframe for generating scatter matrix 

	pd.plotting.scatter_matrix(
								iris_dataframe, 
								c=y_train,
								figsize=(15,15), 
								marker='o', 
								) # generating graph

	plt.show()


if '__main__' == __name__:
	iris_dataset_plotting()