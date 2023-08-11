import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "/home/hanyi/Projects/Frame_Definition_Generation/Llama-2/Finetuned_models/E&FE/meta-llama/Llama-2-7b-hfE&FE"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
question="Question: What is the definition of the word translate in the sentence ' Waffenruhe '' translates as truce , armistice , an uneasy , provisional peace .'? The answer should include following words: 'Source_symbol', 'Source_representation', 'Target_symbol', 'Target_representation', 'Content'. Note that not all these words are necessarily included.Answer:"
#"Question: What is the definition of the word translate in the sentence ' Waffenruhe '' translates as truce , armistice , an uneasy , provisional peace .'? The answer should include following words: 'Source_symbol', 'Source_representation', 'Target_symbol', 'Target_representation', 'Content'. Note that not all these words are necessarily included. Answer:",A  Source_symbol in a Source_representation system is presented to be matched by a Target_symbol in a Target_representation in the ability to express a particular Content.,Have_as_translation_equivalent

inputs  = tokenizer(question,return_tensors="pt")
input_ids = inputs.input_ids.to('cuda') 
model=model.to('cuda')
generate_ids = model.generate(input_ids,max_new_tokens=300,repetition_penalty=1.2)
outputs=tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
raw_outputs=outputs
print(raw_outputs)
           