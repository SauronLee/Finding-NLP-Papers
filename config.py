import json
import os
from tqdm import tqdm
import numpy as np
import argparse
from pretrained_models import T5_QG
#from pretrained_models.BERT_fill_blank import BERT_fill_blanker
import stanza
import nltk
nltk.download('punkt')
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

#-------------------------------START OF CONFIGURATION--------------------------------------#

###### Global Settings
QG_DEVICE = 0  # gpu device to run the QG module
BERT_DEVICE = 0 # gpu device to run the BERT module


#-------------------------------END OF CONFIGURATION--------------------------------------#
# Loading Glove Model
def load_glove_model(File):
    print("Loading Glove Model")
    glove_model = {}
    with open(File,'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded!")
    return glove_model
glove = load_glove_model("./data/glove.6B.300d.txt")

# Stanza NLP object
stanza.download('en')
stanza_nlp = stanza.Pipeline('en', use_gpu = True)
# QG NLP object
print('Loading QG module >>>>>>>>')
qg_nlp = T5_QG.pipeline("question-generation", model='valhalla/t5-base-qg-hl', qg_format="highlight", gpu_index = QG_DEVICE)
print('QG module loaded.')

# Load BERT_fill_blanker
#print('Loading BERT blender >>>>>>>>')
#bert_fill_blank = BERT_fill_blanker(gpu_index = BERT_DEVICE)
#print('BERT blender loaded.')

# Load xlm-roberta-base
print('Loading xlm-roberta-base >>>>>>>>')
xlm_roberta_base_unmasker = pipeline('fill-mask', model='xlm-roberta-base')

# Load xlm-roberta-base-squad2
model_name = "deepset/xlm-roberta-base-squad2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


print('-------- Loading Paper Data --------')

title_list = np.load("./data/paper_data.npy", allow_pickle=True).tolist()[0]
print('Number of title_list: {}'.format(len(title_list)))
author_list = np.load("./data/paper_data.npy", allow_pickle=True).tolist()[1]
print('Number of author_list: {}'.format(len(author_list)))
abstract_list = np.load("./data/paper_data.npy", allow_pickle=True).tolist()[2]
print('Number of abstract_list: {}'.format(len(abstract_list)))
#根据规则构筑的问题 找到答案
ag_with_q_dict = np.load("./data/ag_with_q_text_dict.npy", allow_pickle=True).tolist()
print('Number of ag_with_q_dict: {}'.format(len(ag_with_q_dict)))
ag_with_q_dict_xlm = np.load("./data/ag_with_q_dict_xlm.npy", allow_pickle=True).tolist()
print('Number of ag_with_q_dict: {}'.format(len(ag_with_q_dict_xlm)))
#词典
jargon_defination_dict = np.load("./data/jargon_defination_dict.npy", allow_pickle=True).tolist()
print('Number of jargon_defination_dict: {}'.format(len(jargon_defination_dict)))

paper_keys = np.load("./data/paper_keyphrase_nodes.npy", allow_pickle=True).tolist()
print('Number of paper_keys: {}'.format(len(paper_keys)))
paper_keys_nom = np.load("./data/paper_keyphrase_nodes_nom.npy", allow_pickle=True).tolist()
print('Number of paper_keys_nom: {}'.format(len(paper_keys_nom)))
qg_with_a_dict = np.load("./data/qg_with_a_dict.npy", allow_pickle=True).tolist()
print('Number of qg_with_a_dict: {}'.format(len(qg_with_a_dict)))
qg_without_a_dict = np.load("./data/qg_without_a_dict.npy", allow_pickle=True).tolist()
print('Number of qg_without_a_dict: {}'.format(len(qg_without_a_dict)))

def get2kFrequencyWords(path):
  wiki_words_frequency = {}
  with open(path, 'r') as f:
    for line in f:
      wiki_words_frequency[line.split()[0].strip()] = line.split()[1].strip()
  wiki_words_1k_frequency = [w for (w,f) in list(wiki_words_frequency.items())[:2000]]
  return wiki_words_1k_frequency

wiki_words_2k_frequency = get2kFrequencyWords("./data/enwiki-20190320-words-frequency.txt")



print("-------- End --------")
rules = ["Theories",
             "Theories",
             "Sectors",
             "Issues",
             "Fields",
             "languages",
             "languages",
             "Issues",
             "Issues",
             "Methods",
             "Methods",
             "Methods",
             "Tasks",
             "Tasks",
             "Datasets",
             "Datasets",
             "Achievements",
             "Achievements"]

questions = ["What theories are presented in this paper?",
             "What theories this paper is based on?",
             "What task does this article belong to?",
             "What is the purpose of this article?",
             "What field does this article belong to?",
             "What language is the data for this article based on?",
             "What languages are used in this article?",
             "What problem does this article solve?",
             "What is the current unresolved issue?",
             "What methods are used in this article?",
             "What technologies are used in this article?",
             "What new methods are proposed in this article?",
             "What tasks are used in this article?",
             "What new tasks are proposed in this article?",
             "Which datasets are used in this article?",
             "What new datasets are presented in this article?",
             "What are the results of this paper?",
             "What are the conclusions of this paper?"]


question_rules_layman = ["Which paper proposed **** ? | Theories",
                  "Which paper is based on **** ? | Theories",
                  "Which paper is in **** ? | Fields",
                  "Which paper solved **** ? | Issues",
                  "Which paper is in **** ? | Fields",
                  "Which paper used ---- in **** ? | languages Datasets",
                  "Which paper is based on **** ? | languages",
                  "Which paper solved **** ? | Issues",
                  "Which paper solved **** ? | Issues",
                  "Which paper used **** ? | Methods, Which paper used **** to solve ---- ? | Methods Issues, Which paper used **** in ---- ?  | Methods Fields",
                  "Which paper used **** ? | Methods, Which paper used **** to solve ---- ? | Methods Issues, Which paper used **** in ---- ?  | Methods Fields",
                  "Which paper proposed **** ? | Methods, Which paper used **** in ---- ?  | Methods Fields",
                  "Which paper used **** ? | Tasks, Which paper used **** in ---- ?  | Tasks Fields",
                  "Which paper proposed **** ? | Tasks, Which paper used **** in ---- ?  | Tasks Fields",
                  "Which paper used **** ? | Datasets, Which paper used **** in ---- ? | Datasets Methods",
                  "Which paper proposed **** ? | Datasets",
                  "Which paper results are **** ? | Achievements, \
                  Which paper results are **** in ---- ? | Achievements Datasets, \
                  Which paper results are **** in ---- ? | Achievements languages, \
                  Which paper results are **** in ---- ? | Achievements Tasks, \
                  Which paper results are **** in ---- ? | Achievements Fields, \
                  Which paper results are **** in ---- ? | Achievements Methods",
                  "Which paper results are **** ? | Achievements, \
                  Which paper results are **** in ---- ? | Achievements Datasets, \
                  Which paper results are **** in ---- ? | Achievements languages, \
                  Which paper results are **** in ---- ? | Achievements Tasks, \
                  Which paper results are **** in ---- ? | Achievements Fields, \
                  Which paper results are **** in ---- ? | Achievements Methods"]

question_rules = ["Which paper proposed the theories are **** ? | Theories",
                  "Which paper is based on the theories that **** ? | Theories",
                  "Which paper the field that is **** ? | Fields",
                  "Which paper solved issues are **** ? | Issues",
                  "Which paper the field that is **** ? | Fields",
                  "Which paper used dataset is ---- in **** ? | languages Datasets",
                  "Which paper is based on the language that is **** ? | languages",
                  "Which paper solved issues are **** ? | Issues",
                  "Which paper solved issues are **** ? | Issues",
                  "Which paper used method is **** ? | Methods, Which paper used method is **** to solve issues are ---- ? | Methods Issues, Which paper used method is **** the field that is ---- ?  | Methods Fields",
                  "Which paper used method is **** ? | Methods, Which paper used method is **** to solve issues are ---- ? | Methods Issues, Which paper used method is **** the field that is ---- ?  | Methods Fields",
                  "Which paper proposed method is **** ? | Methods, Which paper used method is **** the field that is ---- ?  | Methods Fields",
                  "Which paper used task is **** ? | Tasks, Which paper used task is **** the field that is ---- ?  | Tasks Fields",
                  "Which paper proposed task is **** ? | Tasks, Which paper used **** the field that is ---- ?  | Tasks Fields",
                  "Which paper used dataset is **** ? | Datasets, Which paper used dataset is **** based on ---- ? | Datasets Methods",
                  "Which paper proposed dataset is **** ? | Datasets",
                  "Which paper results are **** ? | Achievements, \
                  Which paper results are **** based on ---- ? | Achievements Datasets, \
                  Which paper results are **** the language that is ---- ? | Achievements languages, \
                  Which paper results are **** the task that is ---- ? | Achievements Tasks, \
                  Which paper results are **** the field is in ---- ? | Achievements Fields, \
                  Which paper results are **** based on ---- ? | Achievements Methods",
                  "Which paper results are **** ? | Achievements, \
                  Which paper results are **** based on ---- ? | Achievements Datasets, \
                  Which paper results are **** the language that is ---- ? | Achievements languages, \
                  Which paper results are **** the task that is ---- ? | Achievements Tasks, \
                  Which paper results are **** the field that is ---- ? | Achievements Fields, \
                  Which paper results are **** based on ---- ? | Achievements Methods"]

