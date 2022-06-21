"""
voting
"""
import pandas as pd

# index_col 將column 設為index
csv_1 = pd.read_csv(f'./output (1).csv',encoding='utf-8',index_col = 0)
csv_2 = pd.read_csv(f'./output (2).csv',encoding='utf-8',index_col = 0)
csv_3 = pd.read_csv(f'./output (3).csv',encoding='utf-8',index_col = 0)
csv_4 = pd.read_csv(f'./output (4).csv',encoding='utf-8',index_col = 0)
csv_5 = pd.read_csv(f'./output (5).csv',encoding='utf-8',index_col = 0)
csv_6 = pd.read_csv('./output (1) (1).csv',encoding='utf-8',index_col = 0)
csv_7 = pd.read_csv('./output (3) (1).csv',encoding='utf-8',index_col = 0)
# csv_8 = pd.read_csv('./423/result_87.2_ok.csv',encoding='utf-8',index_col = 0)
# csv_9 = pd.read_csv('./423/result_87.0_ok.csv',encoding='utf-8',index_col = 0)
# csv_10 = pd.read_csv('./423/result_86.8_ok.csv',encoding='utf-8',index_col = 0)
# csv_11 = pd.read_csv(f'./423/result_86.5_ok.csv',encoding='utf-8',index_col = 0)
# csv_12 = pd.read_csv(f'./423/result_86.0_ok.csv',encoding='utf-8',index_col = 0)
# csv_13 = pd.read_csv(f'./423/result_85.8_ok.csv',encoding='utf-8',index_col = 0)
# csv_14 = pd.read_csv('./423/result_85.6_ok.csv',encoding='utf-8',index_col = 0)
# csv_15 = pd.read_csv('./423/result_85.3_ok.csv',encoding='utf-8',index_col = 0)
# csv_16 = pd.read_csv('./423/result_85.0_ok.csv',encoding='utf-8',index_col = 0)
#
# csv_17 = pd.read_csv('./426/less_result_84.7_ok.csv',encoding='utf-8',index_col = 0)
# csv_18 = pd.read_csv('./426/less_result_85.3_ok.csv',encoding='utf-8',index_col = 0)
# csv_19 = pd.read_csv('./426/less_result_85.9_ok.csv',encoding='utf-8',index_col = 0)
# csv_20 = pd.read_csv('./426/result_84.3_ok.csv',encoding='utf-8',index_col = 0)
# csv_21 = pd.read_csv('./426/result_84.2_ok.csv',encoding='utf-8',index_col = 0)
# csv_22 = pd.read_csv('./426/result_83.6 (1)_ok.csv',encoding='utf-8',index_col = 0)
#
# csv_23 = pd.read_csv(f'./426/result_82.5_ok.csv',encoding='utf-8',index_col = 0)
# csv_24 = pd.read_csv(f'./426/result_82.9_ok.csv',encoding='utf-8',index_col = 0)
# csv_25 = pd.read_csv(f'./426/result_83.5_ok.csv',encoding='utf-8',index_col = 0)
# csv_26 = pd.read_csv(f'./426/result_83.6_ok.csv',encoding='utf-8',index_col = 0)
# csv_27 = pd.read_csv(f'./426/result_85.2_ok.csv',encoding='utf-8',index_col = 0)

# csv_28 = pd.read_csv(f'./426/roberta_result_82.5_ok.csv',encoding='utf-8',index_col = 0)
# csv_29 = pd.read_csv(f'./426/roberta_result_83.1_ok.csv',encoding='utf-8',index_col = 0)
# csv_30 = pd.read_csv(f'./426/roberta_result_82.5_1_ok.csv',encoding='utf-8',index_col = 0)
# csv_31 = pd.read_csv(f'./426/roberta_result_84.2_ok.csv',encoding='utf-8',index_col = 0)
# csv_32 = pd.read_csv(f'./426/roberta_result_81.9_ok.csv',encoding='utf-8',index_col = 0)
# csv_33 = pd.read_csv(f'./426/roberta_result_81.4_ok.csv',encoding='utf-8',index_col = 0)

# csv_31 = pd.read_csv(f'./427/mac_result_83.1_1_ok.csv',encoding='utf-8',index_col = 0)
# csv_32 = pd.read_csv(f'./427/mac_result_83.1_ok.csv',encoding='utf-8',index_col = 0)
# csv_33 = pd.read_csv(f'./427/mac_result_83.6_ok.csv',encoding='utf-8',index_col = 0)
# csv_31 = pd.read_csv(f'./427/mac_result_84.2_1_ok.csv',encoding='utf-8',index_col = 0)
# csv_32 = pd.read_csv(f'./427/mac_result_84.2_ok.csv',encoding='utf-8',index_col = 0)
# csv_33 = pd.read_csv(f'./427/mac_result_84.7_ok.csv',encoding='utf-8',index_col = 0)
# csv_40 = pd.read_csv(f'./427/mac_result_85.3_ok.csv',encoding='utf-8',index_col = 0)
# csv_41 = pd.read_csv(f'./427/mac_result_86.4_ok.csv',encoding='utf-8',index_col = 0)
# csv_42 = pd.read_csv(f'./427/mac_result_87.6_ok.csv',encoding='utf-8',index_col = 0)


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
concate_data_frame = pd.concat([csv_1,csv_2,csv_3,csv_4,csv_5, csv_6,csv_7],axis=1).reset_index()
print(concate_data_frame)

new_dataframe = pd.DataFrame(columns=['id', 'class'])

new_dataframe['id'] = concate_data_frame['id']
new_dataframe['class'] = concate_data_frame.mode(axis=1).dropna(axis=1).astype('int')
# #
print(new_dataframe)
new_dataframe.to_csv(f'./result_all.csv',index=False)