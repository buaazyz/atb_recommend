import pandas as pd
import numpy as np


#drop the wrong data of the irank is 1 but buy_flag is 0
def drop_useless(df):
	print('before dropping useless , the shape of data is '+ str(df.shape))
	dfu = df[(df.irank==1) & (df.buy_flag == 0)]
	useless = dfu.drop_duplicates(subset = 'buyer_admin_id').buyer_admin_id.values
	print('useless user '+str(len(useless)))

	df1 = df[~df['buyer_admin_id' ].isin(useless)]
	print('after dropping useless , the shape of data is '+ str(df1.shape))

	print('saving new training dataset')
	df1.to_csv('data/newtrain.csv',index=0)
	print('complete')
	return df1

def time_trans(df):
	df['hour']  = df['log_time'].apply(lambda x:int(x[11:13]))
	df['day']   = df['log_time'].apply(lambda x:int(x[8:10]))
	df['month'] = df['log_time'].apply(lambda x:int(x[5:7]))
	return df



def partition(df):
	xx = df[df.country_id == 'xx']
	yy = df[df.country_id == 'yy']
	zz = df[df.country_id == 'zz']

	df=0

	print('xx number:'+str(xx.shape[0])+'  yy number:'+str(yy.shape[0])+'  zz number:'+str(zz.shape[0]))

	xvalid = xx[xx.irank == 1]
	yvalid = yy[yy.irank == 1]
	zvalid = zz[zz.irank == 1]

	xtrain = xx[xx.irank != 1]
	ytrain = yy[yy.irank != 1]
	ztrain = zz[zz.irank != 1]

	print('xvalid number:'+str(xvalid.shape[0])+'  yvalid number:'+str(yvalid.shape[0])+'  zvalid number:'+str(zvalid.shape[0]))

	print('saving partitional dataset ......')

	xtrain.to_csv('data/xtrain.csv',index=0)
	ytrain.to_csv('data/ytrain.csv',index=0)
	ztrain.to_csv('data/ztrain.csv',index=0)

	xvalid.to_csv('data/xvalid.csv',index=0)
	yvalid.to_csv('data/yvalid.csv',index=0)
	zvalid.to_csv('data/zvalid.csv',index=0)

	print('complete')

	return 0

def main():
	df = pd.read_csv('data/r2train.csv')
	df1 = drop_useless(df)
	df1 = time_trans(df1)
	partition(df1)

if __name__ == '__main__':
    main()