



import os

### use tensorflow or sklearn to finish this training


path ='E:/ALL_Char1004/ALL_Char'

first =set()

for rt, dirs, files in os.walk(path):
    for i in files:
        first.add(i[0])
print(first)