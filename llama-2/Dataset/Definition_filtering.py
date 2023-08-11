
from nltk.corpus import framenet as fn
from nltk.corpus.reader.framenet import PrettyList
import csv
from tqdm import tqdm


with open('/home/hanyi/Projects/frame-definition-generation/llama-2/Dataset/Framenet_nltk_dataset.csv','w',encoding='utf-8-sig',newline='')as a:
            writer=csv.writer(a)
            writer.writerow(['exemplar','definition','frame','verb','FE'])
            for lu_idx in tqdm(range(0,40000)):
                try:
                    if fn.lu(lu_idx).POS=='V':
                        for texts in fn.lu(lu_idx).exemplars[:]:
                                FE_sets=[]
                                FE_core_set=[]
                                for FE in [x for x in fn.lu(lu_idx).frame.FE]:
                                        FE_sets.append(FE)
                                
                                for k in fn.lu(lu_idx).frame.FE.keys():
                                         if fn.lu(lu_idx).frame.FE[k]["coreType"]=='Core':
                                               FE_core_set.append(k)
                                        
                                    
                                    
                                    
                                def_list=fn.lu(lu_idx).frame.definition.split(".")
                                def_len=(len(def_list))
                                definition=def_list[0]

           
                                writer.writerow([texts.text.strip("'"),definition,fn.lu(lu_idx).frame.name,fn.lu(lu_idx).name.split('.')[0],FE_core_set])
                    
                except:
                        continue
                                        






       
