import csv
from tqdm import tqdm
import random

#Train_set的例文数量是79175, Test_set的例文数量是3448
path='/home/hanyi/Projects/Frame_Definition_Generation/Dataset/Framenet_Dataset/Framenet_Exemplars.csv'
train_path='/home/hanyi/Projects/Frame_Definition_Generation/Dataset/Framenet_Dataset/Train_set.csv'
test_path='/home/hanyi/Projects/Frame_Definition_Generation/Dataset/Framenet_Dataset/Test_set.csv'
validation_path='/home/hanyi/Projects/Frame_Definition_Generation/Dataset/Framenet_Dataset/Validation_set.csv'
#count the number of frames, we have 643 frames
with open(path,'r',encoding='utf-8-sig')as a:
    reader=csv.reader(a)
    frame_list=[]

    for idx,line in enumerate(reader):
        if idx>0:
            frame_list.append(line[2])
    frame_list = sorted(set(frame_list),key=frame_list.index)
print('frame的总个数为:'+str(len(frame_list)))

random.seed(42)
random.shuffle(frame_list)


train_num=int(len(frame_list)*0.8)
test_num=int(len(frame_list)*0.9)
train_frame=frame_list[:train_num]
validation_frame=frame_list[train_num:test_num]
test_frame=frame_list[test_num:]
train_list_path="/home/hanyi/Projects/Frame_Definition_Generation/Dataset/Framenet_Dataset/train_frame_list.csv"
test_list_path="/home/hanyi/Projects/Frame_Definition_Generation/Dataset/Framenet_Dataset/test_frame_list.csv"
validation_list_path="/home/hanyi/Projects/Frame_Definition_Generation/Dataset/Framenet_Dataset/validation_frame_list.csv"
with open(train_list_path,'w',encoding='utf-8-sig',newline='')as a:
    writer=csv.writer(a)
    writer.writerows([train_frame])
with open(test_list_path,'w',encoding='utf-8-sig',newline='')as b:
    writer=csv.writer(b)
    writer.writerows([test_frame])
with open(validation_list_path,'w',encoding='utf-8-sig',newline='')as c:
    writer=csv.writer(c)
    writer.writerows([validation_frame])

with open(path,'r',encoding='utf-8-sig')as a,open(train_path,'w',encoding='utf-8-sig',newline='')as b,open(test_path,'w',encoding='utf-8-sig',newline='')as c,open(validation_path,'w',encoding='utf-8-sig',newline='')as d:
    reader=csv.reader(a)
    writer_train=csv.writer(b)
    writer_test=csv.writer(c)
    writer_validation=csv.writer(d)
    writer_train.writerow(['Exemplar','Definition','Frame','Verb','FE'])
    writer_test.writerow(['Exemplar','Definition','Frame','Verb','FE'])
    writer_validation.writerow(['Exemplar','Definition','Frame','Verb','FE'])
    train_exemplar_num=0
    test_exemplar_num=0
    validation_exemplar_num=0
    for idx,line in tqdm(enumerate(reader)):
        if idx>0:
            if line[2] in train_frame:
                writer_train.writerow(line)
                train_exemplar_num=train_exemplar_num+1
                continue
            if line[2] in test_frame:
                writer_test.writerow(line)
                test_exemplar_num=test_exemplar_num+1
                continue
            if line[2] in validation_frame:
                writer_validation.writerow(line)
                validation_exemplar_num=validation_exemplar_num+1
                continue

print('Train_set的Frame数量是'+str(train_num)+'例文数量是'+str(train_exemplar_num))
print('Validation_set的Frame数量是'+str(len(frame_list)-test_num)+'例文数量是'+str(validation_exemplar_num))
print('Test_set的Frame数量是'+str(test_num-train_num)+'例文数量是'+str(test_exemplar_num))


'''
with open(path,'r',encoding='utf-8-sig')as a:
    reader=csv.reader(a)
'''
