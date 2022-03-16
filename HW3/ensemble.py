import csv
row1 = []
row2 = []
row3 = []

with open("./submission_83.266.csv", "r") as file:
    rows = csv.reader(file)

    # 以迴圈輸出每一列
    for step, row in enumerate(rows):
        if step == 0:
            continue
        row1.append(row[1])
    file.close()

with open("./submission_87.55.csv", "r") as file:
    rows = csv.reader(file)

    # 以迴圈輸出每一列
    for step, row in enumerate(rows):
        if step == 0:
            continue
        row2.append(row[1])
    file.close()

with open("./submission_85.159.csv", "r") as file:
    rows = csv.reader(file)

    # 以迴圈輸出每一列
    for step, row in enumerate(rows):
        if step == 0:
            continue
        row3.append(row[1])
    file.close()


final_row = []
for i,j,k in zip(row1, row2, row3):
    if (i == j or i == k) and (i != j or i != k):
        final_row.append(i)
    elif j == k and (i != j or i != k):
        final_row.append(j)
    elif i == j and i == k:
        final_row.append(i)
    else:
        final_row.append(j)

print(row1)
print(row2)
print(row3)
print(final_row)


import csv
with open('ensemble_submission.csv', 'w', newline='') as csvfile:

  # 以空白分隔欄位，建立 CSV 檔寫入器
  writer = csv.writer(csvfile)

  writer.writerow(['Id', 'Category'])
  for step, i in enumerate(final_row):
    writer.writerow([step+1, i])


