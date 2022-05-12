import csv
import json


id = []
row1=[]
appear_UNK_index = []
oringin_answer = []
with open("./427/result_all.csv", "r", encoding='utf-8') as file:
    rows = csv.reader(file)

    # 以迴圈輸出每一列
    for step, row in enumerate(rows):
        if step == 0:
            continue
        row1.append(row[1])
        id.append(row[0])
    file.close()

print(row1)


for index, answer in enumerate(row1):
    if answer.find("[UNK]")>=0:
        print(index, answer)
        appear_UNK_index.append(index)
        oringin_answer.append(answer)



right_answer = []
with open("hw7_test.json", "r", encoding="utf-8") as file:
    data = json.load(file)
    for num, index in enumerate(appear_UNK_index):
        print(data["questions"][index]["paragraph_id"])
        print(data["paragraphs"][data["questions"][index]["paragraph_id"]])

        start = oringin_answer[num].find("[")-1
        end = oringin_answer[num].find("]")+1


        print(start)
        print(end)
        #[UNK] 後
        if len(oringin_answer[num])-1 < end :
            print(oringin_answer[num][start])
            print(data["paragraphs"][data["questions"][index]["paragraph_id"]][data["paragraphs"][data["questions"][index]["paragraph_id"]].find(oringin_answer[num][start]) + 1])

            right_answer.append(oringin_answer[num][:start+1]+data["paragraphs"][data["questions"][index]["paragraph_id"]][data["paragraphs"][data["questions"][index]["paragraph_id"]].find(oringin_answer[num][start]) + 1])


        # [UNK] 前
        elif start < 0:

            print(data["paragraphs"][data["questions"][index]["paragraph_id"]][data["paragraphs"][data["questions"][index]["paragraph_id"]].find(oringin_answer[num][end],77) - 1])
            print(oringin_answer[num][end])
            try:
                print(oringin_answer[num][end+1])
                right_answer.append(data["paragraphs"][data["questions"][index]["paragraph_id"]][
                                        data["paragraphs"][data["questions"][index]["paragraph_id"]].find(
                                            oringin_answer[num][end]+oringin_answer[num][end+1]) - 1] + oringin_answer[num][end:])
                continue
            except:
                pass
            right_answer.append(data["paragraphs"][data["questions"][index]["paragraph_id"]][data["paragraphs"][data["questions"][index]["paragraph_id"]].find(oringin_answer[num][end], 77) - 1]+oringin_answer[num][end:])
        else:
            print(oringin_answer[num][start])


            if data["paragraphs"][data["questions"][index]["paragraph_id"]][data["paragraphs"][data["questions"][index]["paragraph_id"]].find(oringin_answer[num][end]) - 1] \
                == data["paragraphs"][data["questions"][index]["paragraph_id"]][data["paragraphs"][data["questions"][index]["paragraph_id"]].find(oringin_answer[num][start]) + 1]:
                print(data["paragraphs"][data["questions"][index]["paragraph_id"]][
                          data["paragraphs"][data["questions"][index]["paragraph_id"]].find(
                              oringin_answer[num][start]) + 1])

                right_answer.append(oringin_answer[num][:start+1]+data["paragraphs"][data["questions"][index]["paragraph_id"]][
                      data["paragraphs"][data["questions"][index]["paragraph_id"]].find(
                          oringin_answer[num][start]) + 1]+oringin_answer[num][end:])
            elif data["paragraphs"][data["questions"][index]["paragraph_id"]][data["paragraphs"][data["questions"][index]["paragraph_id"]].find(oringin_answer[num][end], 40) - 1] \
                == data["paragraphs"][data["questions"][index]["paragraph_id"]][data["paragraphs"][data["questions"][index]["paragraph_id"]].find(oringin_answer[num][start]) + 1]:
                print(data["paragraphs"][data["questions"][index]["paragraph_id"]][
                          data["paragraphs"][data["questions"][index]["paragraph_id"]].find(
                              oringin_answer[num][start]) + 1])

                right_answer.append(
                    oringin_answer[num][:start + 1] + data["paragraphs"][data["questions"][index]["paragraph_id"]][
                        data["paragraphs"][data["questions"][index]["paragraph_id"]].find(
                            oringin_answer[num][start]) + 1] + oringin_answer[num][end:])


                # print("11:", data["paragraphs"][data["questions"][index]["paragraph_id"]][data["paragraphs"][data["questions"][index]["paragraph_id"]].find(oringin_answer[num][end]) - 1])
                # print("22", data["paragraphs"][data["questions"][index]["paragraph_id"]][data["paragraphs"][data["questions"][index]["paragraph_id"]].find(oringin_answer[num][start]) + 1])
            elif data["paragraphs"][data["questions"][index]["paragraph_id"]][data["paragraphs"][data["questions"][index]["paragraph_id"]].find(oringin_answer[num][end], 5) - 1] \
                == data["paragraphs"][data["questions"][index]["paragraph_id"]][data["paragraphs"][data["questions"][index]["paragraph_id"]].find(oringin_answer[num][start], 5) + 1]:
                print(data["paragraphs"][data["questions"][index]["paragraph_id"]][
                          data["paragraphs"][data["questions"][index]["paragraph_id"]].find(
                              oringin_answer[num][end]) - 1])

                right_answer.append(
                    oringin_answer[num][:start + 1] + data["paragraphs"][data["questions"][index]["paragraph_id"]][
                        data["paragraphs"][data["questions"][index]["paragraph_id"]].find(
                            oringin_answer[num][end]) - 1] + oringin_answer[num][end:])

            else:
                print("None")
                right_answer.append(
                    oringin_answer[num][:start + 1] + data["paragraphs"][data["questions"][index]["paragraph_id"]][data["paragraphs"][data["questions"][index]["paragraph_id"]].find(oringin_answer[num][end], 5) - 1]+ data["paragraphs"][data["questions"][index]["paragraph_id"]][data["paragraphs"][data["questions"][index]["paragraph_id"]].find(oringin_answer[num][start], 5) + 1]+ oringin_answer[num][end:])

            print(oringin_answer[num][end])



        print("-----------------")



print(right_answer)
final_row = []
count = 0
for index, i in enumerate(row1):
    if index in appear_UNK_index:
        final_row.append(right_answer[count])
        count+=1
    else:
        final_row.append(i)

print(final_row)

with open('./427/result_all_ok.csv', 'w', newline='', encoding='utf-8') as csvfile:

  # 以空白分隔欄位，建立 CSV 檔寫入器
  writer = csv.writer(csvfile)

  writer.writerow(['ID', 'Answer'])
  for id, i in zip(id, final_row):
    writer.writerow([id, i])


