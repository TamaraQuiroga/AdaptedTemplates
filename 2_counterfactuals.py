import re
import time
from pathlib import Path
import pandas as pd
import json
import spacy
from utils import clean_arrow, check_different, check_name_method, check_name, create_folder
import os

class Counterfactuals:
    def __init__(self, template, domain, llm_model, prompt_id, experiment_name, bias_type, llm_response_col = "response_llm"):
        # Load bias groups and seed words
        with open("data/all_bias_group.json", 'r') as file:
            bias_groups = json.load(file)
        
        self.template = template
        self.prompt_id = prompt_id
        self.domain = domain
        self.llm_model = llm_model
        self.experiment_name = experiment_name
        self.bias_groups = bias_groups[bias_type]

        self.llm_response_col = llm_response_col
        self.df_llm_template = None
        self.df_counterfactuals = None

        self.df_template = pd.read_csv(f"data/Templates/{self.template}.csv")
        if domain in ["tweets","IMDB"]:
            self.df_NOEs = pd.read_csv(f"data/NOEs/{self.domain}.csv")
        
        df_toxic = pd.read_csv("data/NOEs/wikipedia_talks_toxic.csv").drop(columns=["index"])
        df_non_toxic = pd.read_csv("data/NOEs/wikipedia_talks_nontoxic.csv").drop(columns=["index"])
        self.df_NOEs= pd.concat([df_toxic,df_non_toxic],axis=0).reset_index()

    def load_clean_template_df(self):
        """
        Load the original LLM-generated templates, clean the text,
        and ensure that each template contains the expected identity term.
        """
        input_path = (
            f"adaptation_llm/originales/{self.template}/"
            f"{self.prompt_id}/{self.experiment_name}/"
            f"llm_template_{self.llm_model}_{self.domain}.csv"
        )

        df = pd.read_csv(input_path)

        
        df = df.rename(
            columns={f"c_response_{self.llm_model}_{self.prompt_id}": self.llm_response_col})

        # Text normalization and cleaning
        df[self.llm_response_col] = df[self.llm_response_col].apply(lambda x: clean_arrow(x))
        df[self.llm_response_col] = df[self.llm_response_col].apply(lambda x: x.replace("{", "").replace("}", ""))
        df[self.llm_response_col] = df[self.llm_response_col].apply(lambda x: x.replace("1 .", ""))

        # Keep only relevant columns
        df_clean = df[["sentence", "identity_term", self.llm_response_col]]
        df_clean = df_clean.reset_index()

        total_templates = df_clean.shape[0]

        # Check whether the identity term appears in the LLM output
        df_clean["contains_identity"] = df_clean.apply(
            lambda x: check_name(x["identity_term"], x[self.llm_response_col]),
            axis=1
        )
        n_contains_identity = df_clean["contains_identity"].sum()

        # Check whether the generated text differs from the original template
        df_clean["is_different"] = df_clean.apply(
            lambda x: check_different(x["sentence"], x[self.llm_response_col]),
            axis=1
        )
        n_different = df_clean["is_different"].sum()

        print(f"Templates containing identity terms: {n_contains_identity}/{total_templates}")
        print(f"Templates different from original sentence: {n_different}/{total_templates}")

        # If identity term is missing, fall back to the original sentence
        df_clean[self.llm_response_col] = df_clean.apply(
            lambda x: x["sentence"] if x["contains_identity"] == 0 else x[self.llm_response_col],
            axis=1
        )

        # Re-check identity presence
        df_clean["contains_identity"] = df_clean.apply(
            lambda x: check_name(x["identity_term"], x[self.llm_response_col]),
            axis=1
        )

        assert df_clean["contains_identity"].sum() == total_templates

        self.df_llm_template = df_clean

    def replace_identity_term(self, original_term, new_term, text):
        """
        Replace an identity term with a new one, preserving capitalization.
        """
        if new_term.islower() and text.find(original_term) == 0:
            new_term = new_term.capitalize()

        if original_term in text:
            text = text.replace(original_term, new_term)
        if original_term.upper() in text:
            text = text.replace(original_term.upper(), new_term.upper())
        if original_term.lower() in text:
            text = text.replace(original_term.lower(), new_term.lower())
        if original_term.capitalize() in text:
            text = text.replace(original_term.capitalize(), new_term.capitalize())

        return text
           
    def generate_counterfactual_templates(self, df_source, new_identity_term, col):
        """
        Generate counterfactual templates by replacing identity terms.
        """
        data = {}
        data["template"] = df_source.apply(
            lambda x: self.replace_identity_term(
                x["identity_term"], new_identity_term, x[col]
            ),
            axis=1
        )
        data["template_index"] = df_source["index"]

        return pd.DataFrame(data)
    
    def generate_type_counterfactuals(self,df_initial,output_path,type_data="llm"):
        """
        Generate counterfactual templates for all bias groups and seed words.
        """
        all_counterfactual_dfs = []
        if type_data == "llm":
            col = self.llm_response_col
        else:
            col = "sentence"
            if "index" not in df_initial.columns:
                df_initial = df_initial.reset_index()
            if "sentence" not in df_initial.columns:
                df_initial = df_initial.rename(columns={"template":"sentence"})
        for group in self.bias_groups:
            if group != "Female" and not group.startswith("F-"):
                for seed in self.bias_groups[group]["seed_words"]:
                    df_cf = self.generate_counterfactual_templates(df_initial, seed,col)
                    df_cf["group"] = group
                    df_cf["identity_term"] = seed
                    all_counterfactual_dfs.append(df_cf)
            else:
                for seed in self.bias_groups[group]["seed_words"]:
                    df_cf = self.generate_counterfactual_templates(
                        df_initial, seed, col
                    )
                    df_cf["group"] = group
                    df_cf["identity_term"] = seed
                    all_counterfactual_dfs.append(df_cf)

        df_output = pd.concat(all_counterfactual_dfs, axis=0).reset_index(drop=True)

        # Verify that identity terms are correctly used
        df_output["valid"] = df_output.apply(
            lambda x: check_name_method(x["identity_term"], x["template"]),
            axis=1
        )

        print(df_output["valid"].sum(), df_output.shape[0])
        assert df_output["valid"].sum() == df_output.shape[0]

        df_output = df_output.drop(columns=["valid"])
        # self.df_counterfactuals = df_output

        if type_data == "llm":
            df_output["template_type"] = f"{self.llm_model}_{self.domain}"
        elif type_data == "NOEs":
            df_output["template_type"] = f"{self.domain}"
        elif type_data =="template":
            df_output["template_type"] = f"{self.template}"
        print(output_path)
        p = Path(output_path)
        dir_to_create = p if p.suffix == "" else p.parent
        dir_to_create.mkdir(parents=True, exist_ok=True)
        df_output.to_csv(output_path, index=False)
        print(f"Counterfactuals {type_data} file written successfully")

    def generate_all_counterfactuals(self,type_data="llm"):
        output_path = (
                f"adaptation_llm/Counterfactuals/"
                f"{self.template}/{self.prompt_id}/{self.experiment_name}/"
                f"pertubation_{self.llm_model}_{self.domain}.csv")
        if os.path.exists(output_path):
            print( "scores LLMs already exits")
        else:
            self.generate_type_counterfactuals(self.df_llm_template,output_path,"llm")
        output_path = (
                f"adaptation_llm/Counterfactuals/"
                f"{self.template}/"
                f"pertubation_{self.template}.csv"
            )
        if os.path.exists(output_path):
            print( "scores template already exits")
        else:
            self.generate_type_counterfactuals(self.df_template,output_path,"template")
        output_path = (
                f"adaptation_llm/Counterfactuals/NOEs/"
                f"pertubation_{self.domain}.csv")
        if os.path.exists(output_path):
            print( "scores NOEs already exits")
        else:
            self.generate_type_counterfactuals(self.df_NOEs,output_path,"NOEs")



# with open("all_bias_group.json") as f:
#     dict_all_groups = json.load(f)




# parametros
template = "EEC" 
id_prompt = "f8"
experiment_name = "few_shot"
bias = "nationality"

# Crear la carpeta si no existe
# nombre_carpeta = Path(path_folder)
# nombre_carpeta.mkdir(parents=True, exist_ok=True)
path_folder =  f"adaptation_llm/Counterfactuals/{experiment_name}/{template}/{id_prompt}/"
create_folder(path_folder)

col_response_llm = f"response_llm"
for domain in ["tweets","IMDB","wikipedia_talks"]:
    for model_llm in ["llama3_70"]:
            object =Counterfactuals(template,
                                    domain,
                                    model_llm,
                                    id_prompt,
                                    experiment_name, 
                                    bias,
                                    llm_response_col=col_response_llm,
                                    )
            object.load_clean_template_df()
            object.generate_all_counterfactuals()
            #que saque counterfactual del dominio (NOEs)
            # que saque counterfactual del template
