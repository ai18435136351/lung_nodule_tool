import csv
from outline_gen import area, weight

data_mask = csv.reader(open("acc2.csv", 'r'))
name_mask = []
for i in data_mask:
    name_mask.append(i[0])


def volume_count(file1, file2, x, y):
    img_volume = 0
    for i in range(80):
        if file1 + str(i) in name_mask:
            # print(file1 + str(i))
            c = int(area(file1, file2, file1 + str(i), x, y))
            print(c)
            img_volume = img_volume + c
    print('tiji:', img_volume)
    return(img_volume)


def weight_count(file1, file2, x, y):
    img_weight = 0
    for i in range(80):
        if file1 + str(i) in name_mask:
            c = int(weight(file1, file2, file1 + str(i), x, y))
            img_weight = img_weight + c
    print('zhongl:', img_weight)
    return (img_weight)

# volume_count('chenhui', 'chenhui1', 604, 725)
# weight_count('chenhui', 'chenhui1', 604, 725)