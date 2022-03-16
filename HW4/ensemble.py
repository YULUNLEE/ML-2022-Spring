import csv
row1 = []
row2 = []
row3 = []
row4 = []
id = []
with open("./output_ensemble.csv", "r") as file:
    rows = csv.reader(file)

    # 以迴圈輸出每一列
    for step, row in enumerate(rows):
        if step == 0:
            continue
        row1.append(row[1])
        id.append(row[0])
    file.close()

with open("./output_model356_4_91.csv", "r") as file:
    rows = csv.reader(file)

    # 以迴圈輸出每一列
    for step, row in enumerate(rows):
        if step == 0:
            continue
        row2.append(row[1])
    file.close()

with open("./output_model356_1_92.csv", "r") as file:
    rows = csv.reader(file)

    # 以迴圈輸出每一列
    for step, row in enumerate(rows):
        if step == 0:
            continue
        row3.append(row[1])
    file.close()

with open("./ensemble_submission76.csv", "r") as file:
    rows = csv.reader(file)

    # 以迴圈輸出每一列
    for step, row in enumerate(rows):
        if step == 0:
            continue
        row4.append(row[1])
    file.close()


final_row = []
# 三種
for i,j,k in zip(row1, row2, row3):
    if (i == j or i == k) and (i != j or i != k):
        final_row.append(i)
    elif j == k and (i != j or i != k):
        final_row.append(j)
    elif i == j and i == k:
        final_row.append(i)
    else:
        final_row.append(k)

# 四種
# for i,j,k,l in zip(row1, row2, row3, row4):
#     if i == j and (i != k and i != l):
#         final_row.append(i)
#     elif i == k and (i != j and i != l):
#         final_row.append(i)
#     elif i == l and (i != j and i != k):
#         final_row.append(i)
#     elif j == k and (j != l and j != i):
#         final_row.append(j)
#     elif j == l and (j != k and j != i):
#         final_row.append(j)
#     elif k == l and (k != i and k != j):
#         final_row.append(k)
#     elif (i == j and i == k) and (i != l):
#         final_row.append(i)
#     elif (i == j and i == l) and (i != k):
#         final_row.append(i)
#     elif (i == k and i == l) and (i != j):
#         final_row.append(i)
#     elif (j == k and j == l) and (j != i):
#         final_row.append(j)
#     elif (i == j and i == k and i == l):
#         final_row.append(i)
#     else:
#         final_row.append(i)


print(row1)
print(row2)
print(row3)
# print(row4)
print(final_row)


import csv
with open('ensemble_submission.csv', 'w', newline='') as csvfile:

  # 以空白分隔欄位，建立 CSV 檔寫入器
  writer = csv.writer(csvfile)

  writer.writerow(['Id', 'Category'])
  for id, i in zip(id, final_row):
    writer.writerow([id, i])


