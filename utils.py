


import re
import os 
import numpy as np

def remove_parentheses_content(text: str) -> str:
    return re.sub(r'\([^)]*\)', '', text)

def clean_response(llm_text: str) -> str:
    end_text = "without any introduction or explanation."
    len_text = len(end_text)
    index_text = llm_text.find(end_text)+len_text
    if index_text==-1:
        return llm_text
    clean_text = llm_text[index_text:]
    clean_text = clean_text.lstrip()
    first_newline = str(clean_text).rstrip().strip().find("\n")
    if first_newline>0:
        clean_text = clean_text[:first_newline]
    indice = max([clean_text.find("->"),clean_text.find("1.")])
    if indice != -1:
        clean_text =  clean_text[indice:]
    clean_text = clean_text.replace("[/INST]","")
    clean_text = clean_text.replace('"','')
    if "domain" in clean_text or "Domain" in clean_text or "task" in clean_text or "tweet" in clean_text or "Tweet" in clean_text or "Twitter" in clean_text or "twitter" in clean_text or "Text" in clean_text or "text" in clean_text or "wiki" in clean_text or "Wiki" in clean_text or "IMDB" in clean_text or "emotion" in clean_text or "semanti" in clean_text:
        clean_text = ""
    if len(clean_text.split(" "))<4:
        clean_text = ""
    clean_text = remove_parentheses_content(clean_text)
    return clean_text

def text_contains_name(name: str,llm_text: str) -> bool:
    llm_text = str(llm_text)
    if name in llm_text or name.upper() in llm_text or name.lower() in llm_text or name.capitalize() in llm_text:
        return True
    return False

def text_different_from_template(template: str,response_template: str) -> bool:
    l = response_template.split(" ")
    if template.rstrip() == response_template.rstrip() or response_template.strip()== template.strip():
        return False
    elif template.replace(".","").capitalize() == response_template.replace(".","").capitalize():
        return False
    elif template.replace(",","").replace(".","").replace("!","").rstrip().strip() == response_template.replace(",","").replace(".","").replace("!","").rstrip().strip()  :
        return False
    return True

def few_shot_format(x,y):
    return f"'{y}' is rewritten as {x}'"


def create_folder(folder_name):
    try:
        os.makedirs(folder_name, exist_ok=True)  # crea padres también
        print(f"Folder '{folder_name}' ready.")
    except OSError as e:
        print(f"Error creating folder: {e}")

def check_name(name,llm_text):
    llm_text = str(llm_text)
    if name in llm_text or name.upper() in llm_text or name.lower() in llm_text or name.capitalize() in llm_text:
        return 1
    return 0

def check_name(name,llm_text):
    llm_text = str(llm_text)
    if name in llm_text or name.upper() in llm_text or name.lower() in llm_text or name.capitalize() in llm_text:
        return 1
    return 0


def check_name_method(name,sentence):
    if name in sentence  or name.upper() in sentence or name.lower() in sentence or name.capitalize() in sentence:
        return 1     
    return 0

def check_different(template,response_template):
    l = response_template.split(" ")
    if template.rstrip() == response_template.rstrip() or response_template.strip()== template.strip():
        return 0
    elif template.replace(".","").capitalize() == response_template.replace(".","").capitalize():
        return 0
    elif template.replace(",","").replace(".","").replace("!","").rstrip().strip() == response_template.replace(",","").replace(".","").replace("!","").rstrip().strip()  :
        return 0
    return 1

def clean_arrow(x):
    index_arrow = x.find("->")
    if index_arrow !=-1:
        new = x[index_arrow+2:]
        return new
    return x


def preprocess(text):
    new_text = []
    text = str(text)
    if len(text)>0:
        for t in text.split(" "):
            #t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)
    else:
        return ""
    
def index_word(text,word):
    index_word = text.find(word)
    return text[index_word+len(word):]


def MAE(a, b):
    v = np.mean(np.abs(a-b))
    return v

def MSE( a, b):
    v = np.mean((a-b)**2)
    return v

def clean_unnamed(df):
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df = df.reset_index(drop=True) 
    return df   