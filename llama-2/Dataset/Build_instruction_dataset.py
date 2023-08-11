import csv
import tqdm
import re
mode='test'
pattern='E&FE' #E,E&FE,E,Z
if mode=='train':  
    input_path='/home/hanyi/Projects/Frame_Definition_Generation/Dataset/Framenet_Dataset/Train_set.csv'
    output_path='/home/hanyi/Projects/Frame_Definition_Generation/Llama-2/Dataset/'+'Train_'+pattern+'.csv'
if mode=='test':
    input_path='/home/hanyi/Projects/Frame_Definition_Generation/Dataset/Framenet_Dataset/Test_set.csv'
    output_path='/home/hanyi/Projects/Frame_Definition_Generation/Llama-2/Dataset/'+'Test_'+pattern+'.csv'
with open(input_path,'r',encoding='utf-8-sig')as a, open(output_path,'w',encoding='utf-8-sig', newline='')as b:
    reader=csv.reader(a)
    writer=csv.writer(b)
    writer.writerow(['text'])
    for idx,line in enumerate(reader):
        word_list=line[4].strip('[').strip(']')
        if idx>0 and mode=='train':
            if pattern=='E':
                prompt = "Question: What is the definition of the word "+line[3]+" in the sentence '"+line[0]+" Answer:" + line[1]
            if pattern=='E&FE':
                 prompt = "Question: What is the definition of the word "+line[3]+" in the sentence '"+line[0]+"'? The answer should include following words: " + word_list+ ". Note that not all these words are necessarily included. Answer:" + line[1]

            writer.writerow([prompt])
        if idx>0 and mode=='test':
            if pattern=='E':
                prompt =  "Question: What is the definition of the word "+line[3]+" in the sentence '"+line[0]+" Answer:" 
            if pattern=='E&FE':
                word_list=line[4].strip('[').strip(']')
                prompt =  "Question: What is the definition of the word "+line[3]+" in the sentence '"+line[0]+"'? The answer should include following words: " + word_list+ ". Note that not all these words are necessarily included. Answer:" 

            
            writer.writerow([prompt,line[1],line[2]])
