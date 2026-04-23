

import subprocess
from tqdm import tqdm 
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import operator
import gc
from pathlib import Path
import json
import time 
import random
import re
from utils import remove_parentheses_content, clean_response, text_contains_name ,text_different_from_template, few_shot_format, create_folder
import sys, subprocess
from dotenv import load_dotenv

load_dotenv()

command = [
    sys.executable, "-m", "huggingface_hub.cli",
    "login", "--token", "TU_TOKEN_AQUI", "--add-to-git-credential"
]
result = subprocess.run(command, capture_output=True, text=True)
print(result.stdout, result.stderr)

# result = subprocess.run(command, capture_output=True, text=True)
# if result.returncode == 0:
#     print("Login exitoso:")
#     print(result.stdout)
# else:
#     print("Error en el login:")
#     print(result.stderr)



class AdaptedTemplateLLM:
    def __init__(self,
                template_ : str,
                domain_: str,
                model_llm: str,
                id_prompt: str,
                path_few_examples = None ,
                path_names="data/Names/name_male_popular.csv", 
                name_experiment = "",
                few_examples = False) -> None:

        self.template_ = template_ # str, nombre = EEC, IPTTS
        self.domain_ = domain_ # str, nombre: Wikipedia Talks, Tweets
        self.model_llm = model_llm # str, nombre modelo: LLama3-8, LLama3-70, Mixtral
        self.id_prompt = id_prompt # csv, filas: id, prompt.
        self.few_examples = few_examples

        self.path_template =  f"data/Templates/{template}.csv" # csv, filas: sentence, person, toxicity(si lo requiere)
        self.path_domain_examples = f"data/Domain_Examples/{domain}.csv"
        self.path_names = path_names # csv, filas: firstnames
        self.path_few_examples = path_few_examples 

        self.df_template = None
        self.text_prompt = None
        self.df_domain_examples = None

        self.model = None 
        self.tokenizer = None

        self.name_experiment= name_experiment #str, nombre para guardarlo en un lugar especifico

    def load_model(self, model_llm_input):
        if model_llm_input == "llama3_8":
            model_id = "meta-llama/Llama-3.1-8B-Instruct"
            quant = BitsAndBytesConfig(load_in_8bit=True)
            device_map = {"": 0}  # GPU 0

        elif model_llm_input == "llama3_70":
            model_id = "meta-llama/Llama-3.3-70B-Instruct"
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            device_map = "auto"

        elif model_llm_input == "mixtral":
            model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            device_map = "auto"

        else:
            raise ValueError("only available for: llama3_8, llama3_70 or mixtral models")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        # (opcional) algunos modelos no traen pad_token configurado
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant,
            device_map=device_map,
            torch_dtype=torch.bfloat16,   # puedes cambiar a float16 si tu GPU no soporta bf16
        )

        self.model.eval()
  
    def response_model(self,tokenizer,model,sentence,torch_device,N_mean,do_sample = True, temperature = 0.01)-> str:
        inputs = tokenizer(sentence, return_tensors="pt").to(torch_device)
        outputs = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[-1]+N_mean+10,  
            do_sample=do_sample, 
            temperature=temperature, 
            pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def load_df_domain_examples(self):
        self.df_domain_examples = pd.read_csv(self.path_domain_examples)
        self.df_domain_examples = self.df_domain_examples.drop_duplicates()
        self.df_domain_examples_toxic = self.df_domain_examples[self.df_domain_examples["toxicity"]==1]
        self.df_domain_examples_nontoxic = self.df_domain_examples[self.df_domain_examples["toxicity"]==0]
        
    def load_df_prompt(self):
        with open('data/Prompts/prompts_text.json') as f:
            json_prompts = json.load(f)
        self.text_prompt = json_prompts[self.id_prompt]    

    def load_df_template(self):
        df_template = pd.read_csv(self.path_template)
        if self.template_ == "EEC":
            df_template = df_template[['sentence', 'identity_term',"sa"]]
        else:
            df_template = df_template[['sentence', 'identity_term',"toxicity",]]
        self.df_template = df_template#[:10]
        self.df_template = self.df_template.reset_index(drop=True)

    def get_few_examples(self,sample_size=5):
        with open(self.path_few_examples) as f:
            dict_examples = json.load(f)
        l_few_examples = [few_shot_format(dict_examples[sentence],sentence) for sentence in dict_examples]
        random_sample = random.sample(l_few_examples, sample_size)
        return random_sample
    
    def change_examples(self,n_examples,label=None): 
        if label !="toxic" or (self.df_domain_examples_toxic.shape[0]==0):
            ejemplos_nuevos = "\n".join(  str(k+1)+". "+text for k,text in enumerate(self.df_domain_examples_nontoxic.sample(n=n_examples)["example"].tolist())) 
        else:
            ejemplos_nuevos = "\n".join(  str(k+1)+". "+text for k,text in enumerate(self.df_domain_examples_toxic.sample(n=n_examples)["example"].tolist())) 
        return ejemplos_nuevos
    
    def multi_names(self,n_max=200):
        df_names = pd.read_csv(self.path_names)
        if df_names.shape[0]>=n_max:
            n_max_names = n_max #df_names.shape[0]
        else:
            n_max_names = df_names.shape[0]
        if self.df_template.shape[0]<n_max_names:
            n_max_names =self.df_template.shape[0]
        n_repetition = int(self.df_template.shape[0]/n_max_names)
        l_names = df_names["firstname"].tolist()[:n_max_names]
        delta = self.df_template.shape[0]%n_max_names
        l_names = l_names*n_repetition
        if delta != 0:
            l_names = l_names +l_names[:delta]
        self.df_template["new_names"] = l_names
        self.df_template["sentence"] = self.df_template.apply(lambda x: x["sentence"].replace(x["identity_term"], x["new_names"]) , axis =1 )
        self.df_template["identity_term"] = l_names
        self.df_template = self.df_template.drop(columns=["new_names"])

    def load_df_prompt_text(self, 
                            n_examples = 15, 
                            n_few_examples = 5,
                            ):
        dict_domain_name = {"wikipedia_talks":"Wikipedia Talks Pages",
                            "tweets":"Twitter",
                            "IMDB":"IMDb"}
        #New column with final prompt 
        name_column_prompt = "prompt_"+str(self.id_prompt)
        self.df_template[name_column_prompt] = self.df_template["sentence"].apply(lambda x: self.text_prompt.replace("TEMPLATE_ECC",x))
        self.df_template[name_column_prompt] = self.df_template.apply(lambda x: x[name_column_prompt].replace("XXXX",x["identity_term"]),axis=1)
        if n_examples >0:
            self.df_template[name_column_prompt] = self.df_template.apply(lambda x: x[name_column_prompt].replace("NNN",str(n_examples)),axis=1)
        else:
            self.df_template[name_column_prompt] = self.df_template.apply(lambda x: x[name_column_prompt].replace("DOMAIN",dict_domain_name[self.domain_]),axis=1)

        #Include Domain Examples 
        if self.template_=="IPTTS": 
            examples_non_toxic = [ "\n".join(  str(k+1)+". "+text for k,text in enumerate(self.df_domain_examples_nontoxic.sample(n=n_examples)["example"].tolist())) for i in range(self.df_template.shape[0])]
            if self.df_domain_examples_toxic.shape[0]!=0 :
                examples_toxic = [ "\n".join(  str(k+1)+". "+text for k,text in enumerate(self.df_domain_examples_toxic.sample(n=n_examples)["example"].tolist())) for i in range(self.df_template.shape[0])]
            else:     
                examples_toxic = examples_non_toxic
            examples_tweets = []
            for i, label in enumerate(self.df_template["toxicity"]):
                if label =="toxic" :
                    examples_tweets.append(examples_toxic[i])
                else:      
                    examples_tweets.append(examples_non_toxic[i])
        else:
            examples_tweets = [ "\n".join(  str(k+1)+". "+text for k,text in enumerate(self.df_domain_examples_nontoxic.sample(n=n_examples)["example"].tolist())) for i in range(self.df_template.shape[0])]

        self.df_template["examples_prompt"] = examples_tweets                    
        self.df_template[name_column_prompt] = self.df_template.apply(lambda x: x[name_column_prompt].replace("YYYY",x["examples_prompt"]), axis=1)
        
        #Include Few Examples when there is few-shot
        if self.few_examples:
            examples_few = "\n".join(self.get_few_examples(sample_size=n_few_examples))
            self.df_template[name_column_prompt] = self.df_template.apply(lambda x: x[name_column_prompt].replace("EEEE",examples_few), axis=1)

        #Mixtral Format
        if self.model_llm =="mixtral":
                self.df_template[name_column_prompt] =self.df_template[name_column_prompt].apply(lambda x: "<s> [INST]"+ x +"[/INST]")
        return name_column_prompt

    def n_tokens_generate(self, torch_device):
        #calcular cuantos tokens debe generar
        self.df_domain_examples["size"] = self.df_domain_examples["example"].apply(lambda x : self.tokenizer(x, return_tensors="pt").to(torch_device)["input_ids"].shape[-1])
        N_mean = int(sum( self.df_domain_examples["size"].tolist())/len( self.df_domain_examples["size"].tolist()))+2
        print(f"Numero de tokens a generar es {N_mean}")
        print("--------------")
        return N_mean
    def generate_template(self,
                          n_examples = 15, 
                          n_max_times=5,
                          n_few_examples = 5):
        #Create prompts
        name_column_prompt = self.load_df_prompt_text(n_examples = n_examples,n_few_examples = n_few_examples)
        
        # Save prompts 
        create_folder("adaptation_llm/Check")
        create_folder(f"adaptation_llm/originales/{self.name_experiment}/{self.template_}/{self.id_prompt}")
        self.df_template.to_csv(f"adaptation_llm/Check/df_Template_{self.template_}_{self.domain_}_{self.id_prompt}.csv",index=False)
        print(f"adaptation_llm/Check/df_Template_{self.template_}_{self.domain_}_{self.id_prompt}.csv")
        print("df_Template_check dice: que lo va a preguntar a LLM")
        # Load model
        torch_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.load_model(self.model_llm)
        
        N_tokens = self.n_tokens_generate(torch_device)

        #Ask LLM

        start = time.perf_counter()
        l_response_llm = []
        l_last_prompts = []
        with open(f"adaptation_llm/originales/{self.name_experiment}/{self.template_}/{self.id_prompt}/PRUEBA_template_{self.template_}_{model_llm}_{domain}_{self.id_prompt}.csv","w") as f:
            
            for i, row in tqdm(self.df_template.iterrows(), total=self.df_template.shape[0]):
                name = row["identity_term"]
                prompt = row[name_column_prompt]
                template = row["sentence"]
                label_sentence = None
                if "toxicity" in self.df_template.columns:
                    label_sentence = row["toxicity"]
                old_examples = row["examples_prompt"]
                resp = self.response_model(self.tokenizer,self.model,prompt,torch_device,N_tokens)
                max_times = n_max_times
                while max_times>0:
                    clean_resp = clean_response(resp)
                    is_different = text_different_from_template(template,clean_resp)
                    has_name = text_contains_name(name,clean_resp)
                    if is_different and has_name:
                        l_last_prompts += [prompt]
                        break
                    else:
                        new_examples = self.change_examples(n_examples=n_examples,label=label_sentence)
                        prompt = prompt.replace(old_examples,new_examples)
                        old_examples = new_examples
                        resp = self.response_model(self.tokenizer,self.model,prompt,torch_device,N_tokens)
                        max_times-=1
                if max_times==0:
                    if is_different==0 or has_name==0:
                        clean_resp = template
                        l_last_prompts += [prompt]
                if i == 0:
                    print(prompt)
                    print(clean_resp)                
                f.write(clean_resp+"\n")
                l_response_llm.append(clean_resp)  

 
        # Include new cols in dataframe
        self.df_template["response_"+name_column_prompt.replace("prompt_","")] = l_response_llm
        self.df_template[name_column_prompt+"_final"] = l_last_prompts
        self.df_template.to_csv(f"adaptation_llm/originales/{self.name_experiment}/{self.template_}/{self.id_prompt}/llm_template_{self.model_llm}_{self.domain_}.csv") # PATH        
        end = time.perf_counter()
        with open(f"adaptation_llm/originales/{self.name_experiment}/{self.template_}/{self.id_prompt}/time.csv","a") as ftime:
            ftime.write(f"{self.name_experiment}, {self.model_llm}, {self.domain_}, {self.template_}, {self.id_prompt}, {str(end - start)}\n")


#########



##########
name_experiment = "domain_name"
# Save timestamp
# start = time.time()
gc.collect()
torch.cuda.empty_cache()

# with open(path_time,"w") as f:
for prompt_id in ["f10"]:
    for template in ["EEC"]:
        for model_llm in ["llama3_8"]:
            for domain in ["tweets","wikipedia_talks","IMDB"]:

                print(domain,model_llm)
                data = AdaptedTemplateLLM(  template, #strECC, IPTTS
                                            domain, #str
                                            model_llm,#LLM
                                            id_prompt=prompt_id, # JSON mejor
                                            name_experiment =name_experiment,
                                            # path_few_examples=f"data/Few_Shot_Examples/{domain}_chatgpt_EEC.json",
                                            few_examples = False)  
                
                print(f"{template}_{model_llm}_{domain}_{prompt_id}")
                # nombre_carpeta = Path(f"adaptation_llm/originales/{template}/{prompt_id}/{name_experiment}/")
                # print(nombre_carpeta)
                # nombre_carpeta.mkdir(parents=True, exist_ok=True)

                data.load_df_template()
                data.load_df_prompt()
                data.load_df_domain_examples()
                data.multi_names()
                data.generate_template(n_examples=0)
                # data.df_template.to_csv(f"adaptation_llm/originales/{template}/{prompt_id}/{name_experiment}/llm_template_{model_llm}_{domain}.csv") # PATH        

                # data.load_df_complete(f"/home/tquiroga/llm_test_1/Adaptation/adaptation_llm/originales/{template}/{prompt_id}/{name_experiment}/llm_template_{model_llm}_{domain}_few.csv")
                # data.generate_template(n_examples =n_examples_,prompt_already = f"prompt_{model_llm}_few_f3_final")
                # data.df_template.to_csv(f"adaptation_llm/originales/{template}/{prompt_id}
                # /665557/{name_experiment}/llm_template_{model_llm}_{domain}.csv") # PATH        
                # end = time.time()
                # f.write(f"{model_llm}, {domain}, {template}, {prompt_id}, {str(end - start)}")
                gc.collect()
                torch.cuda.empty_cache()


    #####
    
# print(end - start)
