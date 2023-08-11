import csv
from tqdm import tqdm
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
pattern='zero-shot' # 'E&FE' 'E' 'zero-shot'
path='/home/hanyi/Projects/Frame_Definition_Generation/Dataset/Framenet_Dataset/Framenet_Exemplars.csv'
path_list='/home/hanyi/Projects/Frame_Definition_Generation/Dataset/Framenet_Dataset/test_frame_list.csv'
path_result='/home/hanyi/Projects/Frame_Definition_Generation/Llama-2/Results/Result_'+pattern+".csv"
path_write='/home/hanyi/Projects/Frame_Definition_Generation/Llama-2/Test/Generated_definition_'+pattern+".csv"
'''
with open(path,'r',encoding='utf-8-sig')as a,open(path_list,'r',encoding='utf-8-sig')as b,open(path_result,'r',encoding='utf-8-sig')as c,open(path_write,'w',encoding='utf-8-sig',newline='')as d:
        reader_list=csv.reader(b)
        reader_all=csv.reader(a)
        reader_result=csv.reader(c)
        writer=csv.writer(d)
        frame_list=[]
        FD_list=[]
        for idx, line in enumerate(reader_list):
            if idx==0:
                frame_list=line
        for frame in frame_list:
            for idx1,line1 in enumerate(reader_all):
                if frame == line1[2]:
                    FD_list.append([frame,line1[1]])
                    break
        print('共有以下数量的frame:'+str(len(FD_list)))
        
        for idx2, line2 in enumerate(reader_result):
            
             for FD in FD_list:
                  if FD[1]==line2[2]:
                       FD.append(line2[1].strip("Answer:"))
       
'''       
path_result='/home/hanyi/Projects/Frame_Definition_Generation/Llama-2/Results/Result_'+pattern+".csv"
path_write='/home/hanyi/Projects/Frame_Definition_Generation/Llama-2/Test/Generated_definition_'+pattern+".csv"
path_list='/home/hanyi/Projects/Frame_Definition_Generation/Dataset/Framenet_Dataset/test_frame_list.csv'
with open(path_result,'r',encoding='utf-8-sig')as a,open(path_list,'r',encoding='utf-8-sig')as b,open(path_write,'w',encoding='utf-8-sig',newline='')as c:
        reader_result=csv.reader(a)
        reader_list=csv.reader(b)
        writer=csv.writer(c)
        for idx1,line1 in enumerate(reader_list):
              Frame_list=line1[1:]
              break
        FD_list=[]
        for frame in Frame_list:
                list=[]
                for idx2,line2 in enumerate(reader_result):
                    if line2[3]==frame:
                        list.append(line2[1].replace("Answer:",""))
                        definition=line2[2]
                reader_result=csv.reader(open(path_result,'r',encoding='utf-8-sig'))
                
                FD_list.append([frame,definition,list])
                   
               
                
                
                           

   
        model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
        for FD in FD_list:
            D_embedding = model.encode(FD[1])
            G_embedding_list = model.encode(FD[2])
            
            G_sentence_list=FD[2]
           
          
            average=np.mean(G_embedding_list,0)
         
            
            cos_list=[]
            for G_embedding in G_embedding_list:
                 cos_list.extend(util.pytorch_cos_sim(average,G_embedding).numpy()[0])
            max_index=cos_list.index(max(cos_list))
            print(max_index)
            Max_sentence=G_sentence_list[max_index]
            writer.writerow([FD[0],FD[1],Max_sentence])            