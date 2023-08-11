import csv
import evaluate
import numpy as np
pattern='zero-shot' #E,E&FE,E,Zero-shot
mode='all' # all one

# Python program to get average of a list
def Average(lst):
    return sum(lst) / len(lst)
if mode=='all':
  result_path="/home/hanyi/Projects/Frame_Definition_Generation/Llama-2/Test/Generated_definition_"+pattern+".csv"
  with open(result_path,'r',encoding='utf-8-sig')as a:
    reader=csv.reader(line.replace('/0','') for line in a)
    prediction_list=[]
    reference_list=[]
    for idx, line in enumerate(reader):
      try:
        prediction_list.append(line[1])
        reference_list.append(line[2])
      except:
        continue
    rouge = evaluate.load('rouge')
    rouge_results = rouge.compute(predictions=prediction_list,references=reference_list)
    bertscore = evaluate.load("bertscore")
    bertscore_results = bertscore.compute(predictions=prediction_list, references=reference_list, model_type="distilbert-base-uncased")
    bertscore_list=list(bertscore_results.values())
    bertscore_average=Average(bertscore_list[0])
    print(rouge_results,bertscore_average)

if mode=='one':
  result_path="/home/hanyi/Projects/Frame_Definition_Generation/Llama-2/Test/Generated_definition_"+pattern+".csv"
  with open(result_path,'r',encoding='utf-8-sig')as a:
    reader=csv.reader(line.replace('/0','') for line in a)
    prediction_list=[]
    reference_list=[]
    for idx, line in enumerate(reader):
      try:
        prediction_list.append(line[1])
        reference_list.append(line[2])
      except:
        continue
    rouge = evaluate.load('rouge')
    rouge_results = rouge.compute(predictions=prediction_list,references=reference_list)
    bertscore = evaluate.load("bertscore")
    bertscore_results = bertscore.compute(predictions=prediction_list, references=reference_list, model_type="distilbert-base-uncased")
    bertscore_list=list(bertscore_results.values())
    bertscore_average=Average(bertscore_list[0])
    print(rouge_results,bertscore_average)
    


