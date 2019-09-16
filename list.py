import csv
import cv2
import os

list = []
lists = os.listdir('.')
names = []
for i in range(264):
    list.append([])
    for j in range(1):
        list[i].append([])

for i in range(len(lists)):
    if os.path.splitext(lists[i])[1] == '.png':
        names.append(lists[i])
# print(len(names), names)

for i in range(len(names)):
    for j in range(len(names[i])):
        if names[i][j] == '_':
            list[i][0] = names[i][0:j-5]
        if names[i][j] >= '0' and names[i][j] <= '9':
            list[i][0] = list[i][0] + names[i][j]
label = ['name']

with open("acc2.csv", 'w', newline='') as t:  # numline是来控制空的行数的
    writer = csv.writer(t)  # 这一步是创建一个csv的写入器（个人理解）
    writer.writerow(label)  # 写入标签
    writer.writerows(list)  # 写入样本数据
