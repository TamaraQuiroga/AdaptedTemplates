import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from utils import clean_unnamed, MAE




class ScoreSum:
    def __init__(self,path_df):
        
        self.df_score = pd.read_csv(path_df)
        self.df_score =self.df_score
        print(self.df_score.shape)
        if "2" in  self.df_score.columns and "0" in  self.df_score.columns and "3"  in self.df_score.columns :
            self.df_score["f1"] = self.df_score.apply(lambda x : x["0"]+x["1"]-x["2"]-x["3"], axis=1)
            max_value = self.df_score["f1"].max()
            min_value = self.df_score["f1"].min()
            self.df_score["f1"] = self.df_score["f1"].apply(lambda x: (x-min_value)/(max_value-min_value))

        elif "2" in  self.df_score.columns and "0" in  self.df_score.columns:
            self.df_score["f1"] = self.df_score.apply(lambda x : x["2"]-x["0"], axis=1)
            max_value = self.df_score["f1"].max()
            min_value = self.df_score["f1"].min()
            self.df_score["f1"] = self.df_score["f1"].apply(lambda x: (x-min_value)/(max_value-min_value))

        else:
            self.df_score["f1"] = self.df_score.apply(lambda x : x["1"], axis=1)


        self.df_score = self.df_score[["domain","model","template_type","group","template_index","f1"]]        
        self.df_background = None
                
        self.df_bias_country_NOEs = None
        self.df_bias_country_others = None
        self.df_bias_MCM = None

    def df_background_cal(self):
        df_mean = self.df_score[["domain","model","template_index","template_type","f1"]].groupby(["domain","model","template_index","template_type"]).mean()
        df_mean = df_mean.reset_index()
        self.df_background = df_mean.rename(columns={"f1":"1_background"})
        print("self.df_background",self.df_background.shape)

    def df_score_DP(self):
        print("self.df_score",self.df_score.shape)
        self.df_score = self.df_score.merge(self.df_background, on = ["template_index","template_type","domain","model"], how="left")
        print("self.df_score",self.df_score.shape)

        self.df_score = self.df_score.reset_index(drop = True)
        self.df_score["DP"] = self.df_score.apply(lambda x: 1-np.abs(x["f1"]-x["1_background"]), axis=1)

    def df_DP_country(self):
            domains = self.df_score["domain"].unique().tolist()
            df_aux0 = self.df_score.copy()

            df_aux = df_aux0[df_aux0["template_type"].isin(domains)]
            df_aux = df_aux.reset_index(drop=True)
            self.df_bias_country_NOEs = df_aux[["group","DP","template_type","domain","model"]].groupby(["domain","model","group","template_type"]).mean() #agrupar nivel de tempalte_id
            self.df_bias_country_NOEs = self.df_bias_country_NOEs.reset_index()
            
            df_aux = df_aux0[~df_aux0["template_type"].isin(domains)]
            df_aux = df_aux.reset_index(drop=True)
            self.df_bias_country_others = df_aux[["group","DP","template_type","domain","model"]].groupby(["domain","model","group","template_type"]).mean() #agrupar nivel de tempalte_id
            self.df_bias_country_others = self.df_bias_country_others.reset_index()
            print("self.df_bias_country_others",self.df_bias_country_others.shape)

    def df_DP_country_MEAN(self):
            df_aux = self.df_score.copy()
            # self.df_score = self.df_score[["domain","model","template_type","group","template_index","f1"]]        

            df_bias_MCM_std = df_aux[["domain","model","template_type","template_index","f1"]].groupby(["domain","model","template_type","template_index"]).agg("std").reset_index() #agrupar nivel de tempalte_id
            df_bias_MCM = df_bias_MCM_std[["domain","model","template_type","f1"]].groupby(["domain","model","template_type"]).agg("mean").reset_index() #agrupar nivel de tempalte_id

            self.df_bias_MCM_std = df_bias_MCM_std
            self.df_bias_MCM = df_bias_MCM
            
            print("self.df_bias_MCM",self.df_bias_MCM.shape)




class BiasScores:
    def __init__(self,dict_test_model,l_domains,l_models_llm,prompt,template,name_experiment,bias_type):
        self.dict_models = dict_test_model
        self.l_domains = l_domains
        self.l_models_llm = l_models_llm
        self.prompt = prompt
        self.template = template
        self.name_experiment = name_experiment
        self.bias_type = bias_type

    def wikipedia_talks_NOE_balance(self,model_downstream):
        df_nontoxic =pd.read_csv(f"Scores/NOEs/wikipedia_talks/IPTTS/{model_downstream}/scores_wikipedia_talks_nontoxic_toxic.csv")
        df_toxic =pd.read_csv(f"Scores/NOEs/wikipedia_talks/IPTTS/{model_downstream}/scores_wikipedia_talks_toxic_toxic.csv")
        n_min = min(int(max(df_toxic["template_index"].unique())),int(max(df_nontoxic["template_index"].unique())))
        df_nontoxic = df_nontoxic[df_nontoxic["template_index"].isin(list(range(n_min)))]
        df_toxic = df_toxic[df_toxic["template_index"].isin(list(range(n_min)))]
        df_toxic["template_index"] = df_toxic["template_index"].apply(lambda x: x+1+n_min)
        df = pd.concat([df_nontoxic, df_toxic])
        df = df.reset_index(drop=True)
        df.to_csv(f"Scores/NOEs/wikipedia_talks/IPTTS/{model_downstream}/scores_wikipedia_talks.csv", index=False)

    def get_domain(self,path):
        for domain in ["tweets","wikipedia_talks","IMDB"]:
            if domain in path:
                return domain
        if "IMBD" in path:
            return "IMDB"
        return ""

    def get_llm(self,path):
        for llm in ["llama3_8","llama3_70","mixtral"]:
            if llm in path:
                return llm
        return self.get_domain(path)
    

    def sum_score_by_model(self, model_downstream):
        name_abrev_model = dict_model_meta_template[self.template][dict_model_ft_non[model_downstream]][model_downstream]
        model_original_name = model_downstream.split("_")[0]
        # path_scores = f"adaptation_llm/Scores/{model_downstream}/{self.template}/LLMs/{self.bias_type}/{self.name_experiment}/scores_{self.template}_{self.prompt}_all.csv"
        # path_scores = f"adaptation_llm/Scores/{model_downstream}/{self.template}/LLMs/{self.bias_type}/scores_{self.template}_{self.prompt}_all.csv"
        path_scores = f"adaptation_llm/Scores/{model_downstream}/{self.template}/LLMs/{self.bias_type}/{self.name_experiment}/scores_{self.template}_{self.prompt}_all.csv"
        path = Path( f"adaptation_llm/Scores/{model_downstream}/{self.template}/LLMs/{self.bias_type}/{self.name_experiment}")
        path.mkdir(parents=True, exist_ok=True)
        paths_templates = [f"adaptation_llm/Scores/{model_downstream}/{self.template}/template/{self.bias_type}/scores_{self.template}.csv"]
        # if self.template == "IPTTS":
        #     self.wikipedia_talks_NOE_balance(model_downstream)

        # paths_noes = [f"adaptation_llm/Scores/{model_downstream}/{self.template}/NOEs/{self.bias_type}/scores_{domain_i}.csv" for domain_i in self.l_domains if domain_i != "IMDB" ]
        # paths_noes += [f"adaptation_llm/Scores/{model_downstream}/{self.template}/NOEs/{self.bias_type}/scores_IMDB_V1.csv"]
        # paths_noes = [f"adaptation_llm/Scores/{model_downstream}/{self.template}/NOEs/{self.bias_type}/scores_{domain_i}_V1.csv" for domain_i in self.l_domains if "tweets" in domain_i or "IMDB" in domain_i  ]
        # paths_noes += [f"adaptation_llm/Scores/{model_downstream}/{self.template}/NOEs/{self.bias_type}/scores_{domain_i}.csv" for domain_i in self.l_domains if "wiki" in domain_i  ]
        paths_noes = [f"adaptation_llm/Scores/{model_downstream}/{self.template}/NOEs/{self.bias_type}/scores_{domain_i}.csv" for domain_i in self.l_domains   ]

        # path_llm = [f"adaptation_llm/Scores/{model_downstream}/{self.template}/LLMs/{self.bias_type}/{self.name_experiment}/scores_{model_ft}_{domain_}.csv" for model_ft in self.l_models_llm for domain_ in self.l_domains]
        path_llm = [f"adaptation_llm/Scores/{model_downstream}/{self.template}/LLMs/{self.bias_type}/scores_{model_ft}_{domain_}.csv" for model_ft in self.l_models_llm for domain_ in self.l_domains]

        l_dfs = []

        for path in paths_templates:
            # print(path)
            print(path)
            df = pd.read_csv(path)
            df = clean_unnamed(df)
            df["model"] = name_abrev_model
            df["template_type"] = f"{self.template}"
            df.to_csv(path,index=False)
            for k, domain_i in enumerate(self.l_domains):
                df = pd.read_csv(path)
                df["domain"] = domain_i
                l_dfs.append(df)

        for path in path_llm+paths_noes:
            # print(path)
            print(path)

            df = pd.read_csv(path)
            df = clean_unnamed(df)
            df["model"] = name_abrev_model
            index_word = path.index("scores_")
            len_word = len("scores_")
            df["domain"] = self.get_domain(path)
            df["template_type"] = self.get_llm(path)
            if  "scores_IMDB.csv" in path:
                
                print("max_IMDB",df["template_index"].max())
                sample_1000 = pd.Series(df["template_index"].unique()).sample(n=1000, random_state=42).tolist()
                # df = df[df["template_index"].isin(sample_1000)].reset_index(drop=True)
                df = df[df["template_index"].isin(sample_1000)].reset_index(drop=True)
                print("IMDB n NOEs",df["template_index"].nunique())
                print("shape IMDB",df.shape)
            if "scores_tweets.csv" in path:
            #     path = path.replace("_tweets.csv","_tweets_V1.csv")
            #                 # print(path)
            #     df = pd.read_csv(path)
            #     df = clean_unnamed(df)
            #     df["model"] = name_abrev_model
            #     index_word = path.index("scores_")
            #     len_word = len("scores_")
            #     df["domain"] = self.get_domain(path)
            #     df["template_type"] = self.get_llm(path)
                # sample_700 = pd.Series(df["template_index"].unique()).sample(n=70, random_state=2).tolist()
                # df = df[df["template_index"].isin(list(range(515)))].reset_index(drop=True)
                # print("max_Tweets",df["template_index"].max())
                print("max_Tweets",df["template_index"].max())
                print("total_Tweets",df["template_index"].nunique())
                print("shape Tweets",df.shape)

            l_dfs.append(df)

        df_all_scores = pd.concat(l_dfs, axis = 0)
        df_all_scores = df_all_scores.drop(columns=["template"])
        df_all_scores.to_csv(path_scores,index=False)

    def VBCM_all_models(self, MCM = False):
        L_VBCM_NOEs = []
        L_VBCM_LLMS_Templates= []
        L_MCM = []
        for model_downstream in self.dict_models[self.template]:
            
            name_abrev_model = dict_model_meta_template[self.template][dict_model_ft_non[model_downstream]][model_downstream]
            model_original_name = model_downstream.split("_")[0]
            print(model_downstream)
            # path_scores = f"adaptation_llm/Scores/{model_downstream}/{self.template}/LLMs/{self.bias_type}/{self.name_experiment}/scores_{self.template}_{self.prompt}_all.csv"
            path_scores = f"adaptation_llm/Scores/{model_downstream}/{self.template}/LLMs/{self.bias_type}/{self.name_experiment}/scores_{self.template}_{self.prompt}_all.csv"

            if "llama" not in model_downstream: #not os.path.exists(path_scores):# or model_downstream in ["llama38B_128","LlaMA3-8B"]:
                self.sum_score_by_model(model_downstream)

            path = Path( f"adaptation_llm/Scores/sum/{self.name_experiment}")
            path.mkdir(parents=True, exist_ok=True)
            if not MCM:            
                path_others = f"adaptation_llm/Scores/sum/{self.name_experiment}/{self.template}_{self.bias_type}_{model_downstream}Others.csv"
                path_NOEs = f"adaptation_llm/Scores/sum/{self.name_experiment}/{self.template}_{self.bias_type}_{model_downstream}_NOEs.csv"
                
                if "llama" not in model_downstream: #not os.path.exists(path_scores) or not os.path.exists(path_NOEs):# or model_downstream in ["llama38B_128","LlaMA3-8B"] :
                    templates = ScoreSum(path_scores)
                    templates.df_background_cal()
                    templates.df_score_DP()
                    templates.df_DP_country()
                    df_others = templates.df_bias_country_others
                    df_NOEs = templates.df_bias_country_NOEs

                else:
                    df_others = pd.read_csv(path_others)
                    df_NOEs = pd.read_csv(path_NOEs)
                L_VBCM_LLMS_Templates.append(df_others)
                L_VBCM_NOEs.append(df_NOEs)

                df_others.to_csv(path_others,index=False)
                df_NOEs.to_csv(path_NOEs,index=False)
            else:
                templates = ScoreSum(path_scores)
                templates.df_DP_country_MEAN()
                df_mcm = templates.df_bias_MCM
                L_MCM.append(df_mcm)               

        if not MCM:

            df_VBCM_NOEs = pd.concat(L_VBCM_NOEs, axis=0)
            df_VBCM_NOEs = df_VBCM_NOEs.reset_index(drop=True)
            df_VBCM_NOEs.to_csv(f"adaptation_llm/Scores/sum/{self.name_experiment}/{self.template}_{self.bias_type}_{self.name_experiment}_NOEs.csv",index=False)
            print("escribio")
            df_VCBM_LLMs_Templates = pd.concat(L_VBCM_LLMS_Templates, axis=0)
            df_VCBM_LLMs_Templates = df_VCBM_LLMs_Templates.reset_index(drop=True)
            df_VCBM_LLMs_Templates.to_csv(f"adaptation_llm/Scores/sum/{self.name_experiment}/{self.template}_{self.bias_type}_{self.name_experiment}_Others.csv",index=False)

        else:
            path_MCM = f"adaptation_llm/Scores/sum/{self.name_experiment}/{self.template}_{self.bias_type}_MCM.csv"
            df_MCM = pd.concat(L_MCM, axis=0)
            df_MCM = df_MCM.reset_index(drop=True)
            df_MCM.to_csv(path_MCM,index=False)
  


class PlotMetric:
    def __init__(self, template, name_experiment, l_domains,l_models_llm,prompt, bias_type):
        self.template = template
        self.name_experiment = name_experiment
        self.l_domains = l_domains
        self.l_models_llm = l_models_llm
        self.prompt = prompt
        self.bias_type = bias_type
        self.df_metric_bias = None

    def calculation_df_metric_bias(self,var):
        df_LLMs_Templates = pd.read_csv(f"adaptation_llm/Scores/sum/{self.name_experiment}/{self.template}_{self.bias_type}_{self.name_experiment}_Others.csv")
        df_NOEs = pd.read_csv(f"adaptation_llm/Scores/sum/{self.name_experiment}/{self.template}_{self.bias_type}_{self.name_experiment}_NOEs.csv")
        print(f"adaptation_llm/Scores/sum/{self.name_experiment}/{self.template}_{self.bias_type}_{self.name_experiment}_Others.csv")
        print(f"adaptation_llm/Scores/sum/{self.name_experiment}/{self.template}_{self.bias_type}_{self.name_experiment}_NOEs.csv")
        df_NOEs["template_type"] = df_NOEs["template_type"].apply(lambda x: x.replace("IMBD","IMDB"))
        df_NOEs["domain"] = df_NOEs["domain"].apply(lambda x: x.replace("IMBD","IMDB"))
        df_LLMs_Templates["template_type"] = df_LLMs_Templates["template_type"].apply(lambda x: x.replace("IMBD","IMDB"))
        df_LLMs_Templates["domain"] = df_LLMs_Templates["domain"].apply(lambda x: x.replace("IMBD","IMDB"))

        df_LLMs_Templates = df_LLMs_Templates.merge(df_NOEs, on =["domain","model","group"],how="left",suffixes=("","_NOEs")) 
        if var == "MAE":
            df_LLMs_Templates[var] =df_LLMs_Templates.apply(lambda x : abs(x["DP"]-x["DP_NOEs"]), axis=1)
        elif var =="Pearson":
            df_LLMs_Templates = df_LLMs_Templates.groupby(["domain","model","template_type"])['DP'].corr(df_LLMs_Templates['DP_NOEs']).reset_index()
            df_LLMs_Templates = df_LLMs_Templates.rename(columns={"DP":"Pearson"})
        else:
            print("var can be MAE or Pearson")
        
        df_metric_bias = df_LLMs_Templates.groupby(["domain","model","template_type"])[var].mean().reset_index()
        dict_metric_bias= {}
        for domain_ in self.l_domains:
            df_sum_i=df_metric_bias[df_metric_bias["domain"]==domain_].reset_index(drop=True)
            df_sum_i = df_sum_i.rename(columns={var:var+"_"+domain_}).drop(columns=["domain"])
            dict_metric_bias[domain_] = df_sum_i

        df_metric_bias = dict_metric_bias[self.l_domains[0]]
        for i in range(len(self.l_domains[1:])):
            df_metric_bias = df_metric_bias.merge(dict_metric_bias[self.l_domains[1:][i]],on=["model","template_type"],how="inner")
            {"llama3_70":"LLaMa3-70B","llama3_8":"LLaMa3-8B","mixtral": "Mixtral8x7B"}
        df_metric_bias["template_type"] = df_metric_bias["template_type"].apply(lambda x: x.replace(f"llama3_70",f"{self.template}-LLaMA3-70B").
                                                                                replace(f"llama3_8",f"{self.template}-LLaMA3-8B").
                                                                                replace(f"mixtral",f"{self.template}-Mixtral8x7B"))
        self.df_metric_bias = df_metric_bias
        return self.df_metric_bias


    def table_sum(self):

        # for template in ["IPTTS","EEC"]:
            # for name_experiment in ["again","again_inter"]:
        L_df = []
        for var in ["MAE","Pearson"]:
            df_metric_ = self.calculation_df_metric_bias(var)

            df_final_template = df_metric_[df_metric_["template_type"]==self.template].drop(columns= ["template_type"])
            df_final_nottemplate = df_metric_[df_metric_["template_type"]!=self.template]

            df_final_nottemplate = df_final_nottemplate.merge(df_final_template, on = "model",suffixes=("","_template"))
            for domain in ["IMDB","tweets","wikipedia_talks"]:
                df_final_nottemplate[f"{domain}"]= abs(df_final_nottemplate[f"{var}_{domain}"])-abs(df_final_nottemplate[f"{var}_{domain}_template"])
                df_final_nottemplate = df_final_nottemplate.drop(columns=[f"{var}_{domain}_template",f"{var}_{domain}"])
            df_final_nottemplate = df_final_nottemplate.drop(columns=["model"])

            resumen_metric =df_final_nottemplate.groupby(["template_type"]).mean().reset_index()
            resumen_metric["metric"] = f"{var}"
            L_df.append(resumen_metric)

        resumen_metrics = pd.concat(L_df, axis = 0)
        resumen_metrics["bias"] = self.bias_type
        resumen_metrics["dataset"] = f"{self.template}"
        resumen_metrics["template_type"] = resumen_metrics["template_type"].apply(lambda x: x.replace(f"{self.template}-",""))
        resumen_metrics = resumen_metrics[["bias","dataset","metric","template_type","IMDB","tweets","wikipedia_talks"]]

        return resumen_metrics
        #         resumen.append(resumen_metrics)

        # df_resumen =pd.concat(resumen,axis=0)
        # df_resumen
    def parameters(self):
        label_map = {
            "EEC": {
                "roberta-xlm":"roberta-xlm",
                "BERT-tweet":"BERT-tweet",
                "multi-BERT":"multi-BERT",
                'distil-BERT': 'Distil-BERT\n(FT)',
                'BERT': 'BERT\n(FT)',
                'cardiffnlp-roberta-xlm': 'Cardiffnlp-twitter-XLM\nbase (FT)',
                'cardiffnlp-roberta': 'Cardiffnlp-twitter\nbase (FT)',
                'cardiffnlp-sentiment': 'Cardiffnlp-sentiment\n(off the shelf)',
                'cardiffnlp-emotion': 'Cardiffnlp-emotion\n(off the shelf)',
                "llama38B":"LlaMA3-8B",
                "llama38B_128":"LlaMA3-8B",
                "Qwen":"Qwen",
                "Mistral":"Mistral"

            },
            "IPTTS": {
                'distil-BERT': 'Distil-BERT\n(FT)',
                'BERT': 'BERT\n(FT)',
                'multi-BERT': 'Multi-BERT\nbase (FT)',
                'cardiffnlp-hate': 'Cardiffnlp-hate\n(off the shelf)',
                'cardiffnlp-offensive': 'Cardiffnlp-offensive\n(off the shelf)',
                "llama38B":"Llama38B",
                "llama38B_128":"LlaMA3-8B",
                "Qwen":"Qwen",
                "Mistral":"Mistral"

            }}

        dict_title_domains = {"wikipedia_talks":"Wikipedia Talks Pages", "tweets":"Tweets","IMBD":"IMDB","IMDB":"IMDB","other":"other"}


        hue_order_all = [f'{self.template}',f'{self.template}-LLaMA3-8B', f'{self.template}-LLaMA3-70B', f'{self.template}-Mixtral8x7B']
        dict_llm_names = {f"{self.template}-LLaMA3-70B":"llama3_70",f"{self.template}-LLaMA3-8B":"llama3_8",f"{self.template}-Mixtral8x7B":"mixtral"}
        l_name_cols = [f'{self.template}',f'{self.template}-LLaMA3-8B',f'{self.template}-LLaMA3-70B',f'{self.template}-Mixtral8x7B','NOEs']

        palette_new = {}
        if self.template=="EEC":
            palette_new[l_name_cols[0]] =  "#74bd6c"
            for i in range(1,len(l_name_cols)):
                palette_new[l_name_cols[i]] = sns.color_palette("Blues", n_colors=20)[-3*i-6]#sns.color_palette("YlGn", n_colors=20)[3*i]
        else:
            palette_new[l_name_cols[0]] = "#5aa37b" #sns.color_palette("Blues", n_colors=8)[3]
            for i in range(1,len(l_name_cols)):
                palette_new[l_name_cols[i]] =sns.light_palette("#6688dd", reverse=True, n_colors=15)[3*i]

        l_models = [model for model in label_map[self.template] if model in self.df_metric_bias["model"].unique()]
        model_order = [m for m in label_map[self.template] if m in l_models]
        new_labels = [label_map[self.template][m] for m in model_order]
        new_labels_1 = [lbl.split("\n")[0] for lbl in new_labels]
        new_labels = [rf"$\mathtt{{{lbl}}}$"+"\n" for lbl in new_labels_1]

        hue_order = [name for name in hue_order_all if name == self.template or dict_llm_names[name] in self.l_models_llm]
        return hue_order, model_order, new_labels, dict_title_domains, palette_new
    
    def bar_graph(self,var):
        hue_order, model_order, new_labels,dict_title_domains, palette_new = self.parameters()
        
        # 📊 Crear figura
        fig, axes = plt.subplots(len(self.l_domains), 1, figsize=(18, 9), sharex=True)
        # Primer y Segundo gráfico
        for i,domain_i in enumerate(self.l_domains):
            sns.barplot(
                data= self.df_metric_bias,
                x='model', y=f'{var}_{domain_i}', hue='template_type',
                ax=axes[i], errorbar=None, palette=palette_new, order=model_order, hue_order = hue_order,width=0.6)
            axes[i].set_ylabel(var, fontsize=18)
            axes[i].set_xticklabels(new_labels, rotation=0, fontsize=17)
            axes[i].set_title('$\mathcal{D}$ = ' +dict_title_domains[domain_i], loc="left",fontsize=22,fontweight='semibold')

            if i < len(self.l_domains)-1:
                axes[i].tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=True)
                axes[i].legend_.remove()
            else:
                # axes[i].legend(loc='upper center', ncol=4, bbox_to_anchor=(0.49, -0.4), fontsize=24 )
                # axes[i].set_xlabel("")  # quitar "model"
                axes[i].legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, -0.4), fontsize=24)
                axes[i].xaxis.set_label_coords(0.5,-0.325)  # (x, y)
            if var == "Pearson":
                axes[i].set_ylim(0, self.df_metric_bias[f"{var}_{domain_i}"].max()+0.01)  # <- ajusta este rango según tus datos

        for ax in axes:
            ax.grid(True, linestyle='--',  axis='y',alpha=0.8)
            for spine in ax.spines.values():
                spine.set_visible(True)       # Asegura que sea visible
                spine.set_linewidth(1.5)      # Grosor del borde
                spine.set_color("black")      # Color del borde

        plt.tight_layout()
        plt.savefig(f'graphs/{self.name_experiment}/{var.capitalize()}_{self.template}_{self.name_experiment}_{self.bias_type}_{self.prompt}.pdf', bbox_inches='tight')
        plt.show()
