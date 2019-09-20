import pandas as pd
import numpy as np

def read_data(name='zz',valid=0):
	'''
		name: the id of country included 'xx','yy' and 'zz',
		and if not, read the total dataset

		valid: if read valid dataset
	'''
	if valid == 1:
		if name == 'xx':
			df_train = pd.read_csv('data/xtrain.csv')
			df_valid = pd.read_csv('data/xvalid.csv')
		elif name == 'yy':
			df_train = pd.read_csv('data/ytrain.csv')
			df_valid = pd.read_csv('data/yvalid.csv')
		elif name == 'zz':
			df_train = pd.read_csv('data/ztrain.csv')
			df_valid = pd.read_csv('data/zvalid.csv')
		else:
			temp = pd.read_csv('data/newtrain.csv')
			df_train = temp[temp.irank != 1]
			df_valid = temp[temp.irank == 1]
		print('reading complete')
		return df_train,df_valid
	else:
		if name == 'xx':
			df_train = pd.read_csv('data/xtrain.csv')
		elif name == 'yy':
			df_train = pd.read_csv('data/ytrain.csv')
		elif name == 'zz':
			df_train = pd.read_csv('data/ztrain.csv')
		else:
			temp = pd.read_csv('data/newtrain.csv')
			df_train = temp[temp.irank != 1]
		print('reading complete')
		return df_train

def read_item():
	return pd.read_csv('data/r2item.csv')


def generate_feature(df, save=1,**kwargs):
	'''
		to generate new feature here and merge into training dataset,
		for example, you can merge the price, the category id, the store id and so on into the dataset
		And other features about the user or the order, you should calculate it first before merging
		
		df: training dataset
		**kwargs is used to add optional variables
	'''


	# method 1: merge item features into training dataset 
	print('generate features from item ......')
	dfitem = kwargs['dfitem']
	item_f = dfitem[['item_id','cate_id','store_id','item_price']]
	df = pd.merge(df,item_f,on='item_id',how='left',sort=False)
	df.fillna(0)

	# you can add other methods here




	if save == 1:
		print('saving dataset after adding features')
		filename = kwargs['filename']
		filename = 'data/features/' + filename
		df.to_csv(filename,index=0)
		print('complete')

	return df


def main():
	df = read_data('zz',valid = 0)
	df_item = read_item()
	generate_feature(df, save=1, filename='zz_additem.csv',dfitem=df_item)

if __name__ == '__main__':
    main()