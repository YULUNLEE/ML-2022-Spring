"""
voting
"""
import pandas as pd

# index_col 將column 設為index
csv_1 = pd.read_csv(f'submission_2000_81.csv',encoding='utf-8',index_col = 0)
csv_2 = pd.read_csv(f'submission.csv',encoding='utf-8',index_col = 0)
csv_3 = pd.read_csv(f'submission_early.csv',encoding='utf-8',index_col = 0)
# csv_4 = pd.read_csv(f'submission_mid.csv',encoding='utf-8',index_col = 0)
# csv_5 = pd.read_csv(f'submission_last.csv',encoding='utf-8',index_col = 0)
# csv_6 = pd.read_csv(f'submission_250.csv',encoding='utf-8',index_col = 0)
# csv_7 = pd.read_csv(f'submission_500.csv',encoding='utf-8',index_col = 0)
# csv_8 = pd.read_csv(f'submission_0519.csv',encoding='utf-8',index_col = 0)
# csv_9 = pd.read_csv(f'submission_79.4_500.csv',encoding='utf-8',index_col = 0)
# csv_10 = pd.read_csv(f'submission_79.4_200.csv',encoding='utf-8',index_col = 0)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
concate_data_frame = pd.concat([csv_1,csv_2,csv_3],axis=1).reset_index()
print(concate_data_frame)

new_dataframe = pd.DataFrame(columns=['id', 'label'])

new_dataframe['id'] = concate_data_frame['id']
new_dataframe['label'] = concate_data_frame.mode(axis=1).dropna(axis=1).astype('int')
# #
print(new_dataframe)
new_dataframe.to_csv(f'ensemble_3.csv',index=False)