import os
import pandas as pd
import csv

dirPath = r'./image/'
file_name_list = [f for f in os.listdir(dirPath) if os.path.isfile(os.path.join(dirPath, f))]
ID_list = []
for data in file_name_list:
    ID_list.append(data.split('-')[2])

count = 0
no_contain = []
with open('web_ID.csv', newline='') as csvfile:
  rows = csv.reader(csvfile)
  for row in rows:
    if(row[0] in ID_list):
        count += 1
    else:
        no_contain.append(row[0])

print(count)
print(no_contain)
print("================")
print('CA0416' in ID_list)
print('AR1302' in ID_list)
print('CA1012' in ID_list)






