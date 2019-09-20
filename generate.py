import pandas as pd
import numpy as np
import math
import heapq
import time
import json
import csv
import lightgbm as lgb



def get_retrieval(uid, user_df, hot_item):


    h1 = []

    temp3 = user_df[user_df['buyer_admin_id'] == uid]
    temp3 = temp3[temp3['buy_flag']==1].reset_index()
    if temp3.shape[0] == 0:
        return hot_item
    create_time = temp3.iloc[0]['log_time']
    h_temp = temp3[temp3['log_time'] == create_time]
    h_temp = list(h_temp['item_id'].values)
    h1 = temp3[temp3['log_time'] == create_time].sort_values('item_price')

    h1 = h1.drop_duplicates(subset=['item_id'], keep='first')
    h1 = list(h1['item_id'].values)


    h2 = temp3[temp3['log_time'] != create_time]
    h2 = list(h2['item_id'].drop_duplicates(keep='first').values)

    temp = np.append(h1,h2)
    temp = np.append(temp, hot_item)

    recall = []

    for i in temp:
        if i not in recall:
            recall.append(i)

    return recall[:30]




def get_hot(df0):
    # df0 = df0[df0['country_id'] != 'yy']
    df0 = df0.drop_duplicates(subset=['buyer_admin_id', 'item_id'], keep='first')
    df1 = df0.groupby('item_id').count()
    df1 = df1.sort_values(by='irank', ascending=False)
    df2 = df1.index.values
    return list(df2[:30])




def main():
    df_test = pd.read_csv('data/r2test.csv')
    # df_test = df_test.sample(100)
    df_item = pd.read_csv('data/r2item.csv')
    test_user = df_test['buyer_admin_id'].drop_duplicates().values
    total = len(test_user) / 100
    print(total)

    count = 0

    df_test = df_test.sort_values('irank')
    to_concat = df_item[['item_id','item_price']]
    df_test = pd.merge(df_test,to_concat,on='item_id',how='left',sort=False)
    df_test = df_test.fillna(0)

    hot_item = get_hot(df_test)

    with open('submission/result0831.csv', 'w', newline='') as f:
        writer = csv.writer(f)

        for u in test_user:
            templist = [int(u)]
            recall = get_retrieval(u, df_test, hot_item)
            templist.extend(recall)
            writer.writerow(templist)
            if (count % 50 == 0):
                print(str(round(count / total, 3)) + '%')
            count += 1
    a = np.arange(31)
    df5 = pd.read_csv('submission/result0831.csv', names=a)
    print(df5.isnull().any())
    print(df5.shape)



if __name__ == '__main__':
    main()
