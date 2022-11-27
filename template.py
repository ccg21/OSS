#PLEASE WRITE THE GITHUB URL BELOW!
#https://github.com/ccg21/OSS
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def load_dataset(dataset_path):
	res = pd.read_csv(dataset_path)
	return res 
	
def dataset_stat(dataset_df):	
	n_feats = dataset_df.shape[1] - 1
	n_class0 = dataset_df[dataset_df['target'] == 0].shape[0]
	n_class1 = dataset_df[dataset_df['target'] == 1].shape[0]
	return n_feats, n_class0, n_class1

def split_dataset(dataset_df, testset_size):
	X = dataset_df.drop(columns = 'target', axis=1)
	Y = dataset_df['target']
	x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = testset_size)
	return x_train, x_test, y_train, y_test

def decision_tree_train_test(x_train, x_test, y_train, y_test):
	dt_cls = DecisionTreeClassifier()
	dt_cls.fit(x_train, y_train)
	y_pred = dt_cls.predict(x_test)
	acc = accuracy_score(y_test, y_pred)
	prec = precision_score(y_test,y_pred)
	recall = recall_score(y_test, y_pred)
	return acc, prec, recall

def random_forest_train_test(x_train, x_test, y_train, y_test):
	dt_cls = RandomForestClassifier()
	dt_cls.fit(x_train, y_train)
	y_pred = dt_cls.predict(x_test)
	acc = accuracy_score(y_test, y_pred)
	prec = precision_score(y_test,y_pred)
	recall = recall_score(y_test, y_pred)
	return acc, prec, recall

def svm_train_test(x_train, x_test, y_train, y_test):
	dt_cls = make_pipeline(
		StandardScaler(),
		SVC()
	)
	dt_cls.fit(x_train, y_train)
	y_pred = dt_cls.predict(x_test)
	acc = accuracy_score(y_test, y_pred)
	prec = precision_score(y_test,y_pred)
	recall = recall_score(y_test, y_pred)
	return acc, prec, recall


def print_performances(acc, prec, recall):
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)

if __name__ == '__main__':
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)
	
	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))
	
	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)