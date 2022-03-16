"""
voting
"""
import pandas as pd

# index_col 將column 設為index
csv_1 = pd.read_csv(f'F:/project/pycharm/ML2/HW3/submission_87.55.csv',encoding='utf-8',index_col = 0)
csv_2 = pd.read_csv(f'F:/project/pycharm/ML2/HW3/submission_85.159.csv',encoding='utf-8',index_col = 0)
csv_3 = pd.read_csv(f'F:/project/pycharm/ML2/HW3/submission_83.266.csv',encoding='utf-8',index_col = 0)
csv_4 = pd.read_csv(f'F:/project/pycharm/ML2/HW3/submission_Val0.83583.csv',encoding='utf-8',index_col = 0)
csv_5 = pd.read_csv(f'F:/project/pycharm/ML2/HW3/submission_Val0.85332.csv',encoding='utf-8',index_col = 0)
# csv_1 = pd.read_csv('./prediction_82.36.csv',encoding='utf-8',index_col = 0)
# csv_2 = pd.read_csv('./prediction_82.2.csv',encoding='utf-8',index_col = 0)
# csv_3 = pd.read_csv('./prediction_82.1.csv',encoding='utf-8',index_col = 0)
# csv_4 = pd.read_csv('./prediction82.22.csv',encoding='utf-8',index_col = 0)
# csv_5 = pd.read_csv('./prediction82.33.csv',encoding='utf-8',index_col = 0)
csv_6 = pd.read_csv(f'F:/project/pycharm/ML2/HW3/submission_Val0.81574.csv',encoding='utf-8',index_col = 0)
csv_7 = pd.read_csv(f'F:/project/pycharm/ML2/HW3/submission_Val0.85537.csv',encoding='utf-8',index_col = 0)
csv_8 = pd.read_csv(f'F:/project/pycharm/ML2/HW3/submission_Val0.86331.csv',encoding='utf-8',index_col = 0)
# csv_14 = pd.read_csv('./prediction(13).csv',encoding='utf-8',index_col = 0)


concate_data_frame = pd.concat([csv_1,csv_2,csv_3,csv_4,csv_5,csv_6,csv_7, csv_8],axis=1).reset_index()
print(concate_data_frame)

new_dataframe = pd.DataFrame(columns=['Id','Category'])

new_dataframe['Id'] = concate_data_frame['Id']
new_dataframe['Category'] = concate_data_frame.mode(axis=1).dropna(axis=1).astype('int')
# #
print(new_dataframe)
new_dataframe.to_csv(f'F:/project/pycharm/ML2/HW3/ensemble_submission.csv',index=False)