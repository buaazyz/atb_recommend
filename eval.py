import pandas as pd
import numpy as np



#---------------- the basic function of reading data ,calculating scores and so on --------------------#

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

def MRRscore(item, relist):
    count = 0
    for i in relist:
        count += 1
        if (i == item):
            return 1 / count
    return 0

def hitting(item, relist):
    # print(relist)
    if item in relist:
        return 1
    return 0

#---------- end --------------#


#---------- the match function and rank function -----------------#

def match(user, dftrain, **kwargs):

	# method1 : get history item both bought or not
	history = dftrain[dftrain.buyer_admin_id==user]
	history = history[history['buy_flag'] ==1].sort_values('irank')

	if history.shape[0] == 0:
		return []

	create_time = history.iloc[0]['log_time']
	h1 = history[history['log_time'] == create_time]

	h1 = h1.drop_duplicates(subset=['item_id'], keep='first')
	h1 = list(h1['item_id'].values)




	# his_item = list(history.item_id.drop_duplicates().values)


	# other method

	return h1
	# return his_item

def rank(recall_list, **kwargs):

	return recall_list




def predict(user, dftrain, **kwargs):
	recall_list = match(user, dftrain)
	rank_list = rank(recall_list)

	return rank_list

#---------- end --------------#



def evaluate(dftrain, dfvalid, samplenum=100, seed=None,type='MRR'):

	# step1: get the user list of we want to predict

	print('begin sample')

	sam_valid = dfvalid.sample(samplenum, random_state=seed)
	sam_user = list(sam_valid.buyer_admin_id.values)


	# step2: get the predict list and evaluate


	print('begin evaluate')
	sclist = []   # to record the MRRscore or hitting 

	for user in sam_user:
		prelist = predict(user, dftrain)
		item  = sam_valid[sam_valid.buyer_admin_id==user].item_id.values[0]

		if type == 'hit':
			sclist.append(hitting(item,prelist))
		else:
			sclist.append(MRRscore(item,prelist))

	print('the score of '+type+' is '+ str(np.mean(sclist)))
	
	return 0










def unit_test():
	df_train,df_valid = read_data('yy',valid=1)

	evaluate(df_train,df_valid, 100, seed = 2, type='hit')

	return 0 




def main():
	unit_test()


if __name__ == '__main__':
    main()