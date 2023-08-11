'''
from transformers import AutoTokenizer, LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")

prompt = "What is Nagoya University"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
'''
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

pattern='FE' #'E&FE''E'
# The model that you want to train from the Hugging Face hub
model_name = "meta-llama/Llama-2-7b-hf"
# The instruction dataset to use

dataset_name =  "/home/hanyi/Projects/Frame_Definition_Generation/Llama-2/Dataset/Train_"+pattern+".csv"


# Fine-tuned model name
new_model = model_name+pattern
#训练样本的个数
traing_num=4000#4000
################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "/home/hanyi/Projects/Frame_Definition_Generation/Llama-2/Finetuned_models/"+pattern+"/"+str(new_model)+"/"

# Number of training epochs
num_train_epochs = 4

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 4
# Batch size per GPU for evaluation
per_device_eval_batch_size = 4
# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1
# Enable gradient checkpointing
gradient_checkpointing = True
# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3
# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4
# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001
# Optimizer to use
optim = "paged_adamw_32bit"
# Learning rate schedule (constant a bit better than cosine)
lr_scheduler_type = "constant"
# Number of training steps (overrides num_train_epochs)
max_steps = -1
# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03
# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True
# Save checkpoint every X updates steps
save_steps = 25
# Log every X updates steps
logging_steps = 25

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = None
# Pack multiple short examples in the same input sequence to increase efficiency
packing = False
# Load the entire model on the GPU 0
device_map = {"": 0}
# Load dataset (you can process it here)
dataset = load_dataset("csv", data_files=dataset_name,split='train').shuffle(seed=42)#randomly select 1000 training data

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

# Train model
trainer.train()

# Save trained model
trainer.model.save_pretrained(output_dir)

# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, output_dir)
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
'''
inputs  = tokenizer("What is the definition of the wordgrowin the sentence 'In the front of the borders where I can reach , I grow my vegetables , annuals and low-growing perennials in patches rather than in rows. The answer should include the word 'grower','food'. Answer:", return_tensors="pt")
input_ids = inputs.input_ids.to('cuda') 
generate_ids = model.generate(input_ids,max_new_tokens=300,repetition_penalty=1.2)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
'''
import csv

test_path='/home/hanyi/Projects/Frame_Definition_Generation/Llama-2/Dataset/Test_'+pattern+'.csv'
result_path='/home/hanyi/Projects/Frame_Definition_Generation/Llama-2/Results/Result_'+pattern+'.csv'
print(result_path)
with open(test_path,'r',encoding='utf-8-sig')as a, open (result_path,'w', encoding='utf-8-sig',newline='')as b:
    reader=csv.reader(a)
    writer=csv.writer(b)
    for idx, line in enumerate(reader):
        if idx>0: #and idx<100:
            question=line[0]
            inputs  = tokenizer(question,return_tensors="pt")
            input_ids = inputs.input_ids.to('cuda') 
            generate_ids = model.generate(input_ids,max_new_tokens=300,repetition_penalty=1.2)
            outputs=tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            raw_outputs=outputs
            for sent in outputs.split('.'):
                if "Answer" in sent:
                    outputs=sent
                    break
            print(outputs)
            writer.writerow([raw_outputs,outputs,line[1]])
            

model.save_pretrained("/home/hanyi/Projects/Frame_Definition_Generation/Llama-2/Finetuned_models/"+str(pattern)+"/"+str(new_model)+"/")
tokenizer.save_pretrained("/home/hanyi/Projects/Frame_Definition_Generation/Llama-2/Finetuned_models/"+str(pattern)+"/"+str(new_model)+"/")

"""
model.save_pretrained('/home/hanyi/Projects/Frame_Definition_Generation/Llama-2/Finetuned_models')
tokenizer.save_pretrained('/home/hanyi/Projects/Frame_Definition_Generation/Llama-2/Finetuned_models')

inputs  = tokenizer("<s>[INST] Define the word 'dip' in the sentence 'The company , which has already dipped its toe into the end-user market with firms such as the Bank of Montreal , Library of Congress , Equitable Life and Safeway , would be happy to win just five to ten large SNA accounts over the next year .", return_tensors="pt")


# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

# Run text generation pipeline with our next model
prompt = "What is a large language model?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] Define the word 'prove' in the sentence 'Saturday night dinner at Keele is always special and this year proved to be no exception .'.[/INST]")
print(result[0]['generated_text'])
"""