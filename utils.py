import json
import os
from tqdm import tqdm
import numpy as np
from config import *
from pretrained_models import *
from config import stanza_nlp
import random
from config import qg_nlp
from transformers import pipeline
xlm_roberta_base_unmasker = pipeline('fill-mask', model='xlm-roberta-base')
import random


"""
ml6team/keyphrase-extraction-kbir-inspec
"""
from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from transformers.pipelines import AggregationStrategy
import numpy as np

# Define keyphrase extraction pipeline
class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, model_outputs):
        results = super().postprocess(
            model_outputs=model_outputs,
            aggregation_strategy=AggregationStrategy.SIMPLE,
        )
        return np.unique([result.get("word").strip() for result in results])

# Load pipeline
print("Load pipeline for ml6team/keyphrase-extraction-kbir-inspec")
model_name = "ml6team/keyphrase-extraction-kbir-inspec"
extractor = KeyphraseExtractionPipeline(model=model_name)



def dictToFreq(key_dic):
  keys_fyc = dict()
  for index, keys in key_dic.items():
    for key in keys:
      if key not in keys_fyc.keys():
        keys_fyc[key] = 0
      keys_fyc[key] += 1
  word_freq_sorted = sorted(keys_fyc.items(), key = lambda kv:(kv[1], kv[0]))
  print("length of word_freq:{}".format(len(word_freq_sorted)))
  return word_freq_sorted


#命名正规化：用长短语替换掉短短语

def keysNom(set_form):
  long_str=','.join(set_form)
  clear_key = set()
  for k in set_form:
    if k in long_str: num_k = long_str.count(k)
    if num_k == 1: clear_key.add(k)
  return clear_key

def procsKeys(paper_keys,title_list,keys_fyc,limit):
  keys_fyc = {k:v for (k,v) in keys_fyc}
  key_dic_n = {}
  for i, doc in tqdm(enumerate(paper_keys)):
    unclear_keys = set()
    for key in keys_fyc.keys():
      if key in doc:
        unclear_keys.add(key)
      if key in title_list[i]:
        unclear_keys.add(key)
    key_dic_n[i] = keysNom(unclear_keys)
    key_dic_n = key_dic_n[-(round(len(key_dic_n)*limit)):]
  return key_dic_n

#生成问题
def QG(key_dic, doc_list):
  qg_without_answer_dict = {}
  qg_with_answer_text_dict = {}
  for i, doc in tqdm(enumerate(doc_list)):
    qg_without_answer_dict[i] = nlp.qg_without_answer(doc)
    if i in key_dic.keys():
      for key in key_dic[i]:
        qg_with_answer_text_dict[i] = nlp.qg_with_answer_text(doc, key)
  return qg_without_answer_dict, qg_with_answer_text_dict

def QGT5WithAnswer(key_dic, doc_list):
  qg_with_answer_text_dict = {}
  for paper_i, doc in tqdm(enumerate(doc_list)):
    if paper_i in key_dic.keys():
      for key in key_dic[paper_i]:
        if paper_i not in qg_with_answer_text_dict.keys():
          qg_with_answer_text_dict[paper_i] = []
        qg_with_answer_text_dict[paper_i].append(qg_nlp.qg_with_answer_text(doc, key))
  return qg_with_answer_text_dict

#过滤掉常见词
def keysFilterByFrequencyWords(wiki_words_1k_frequency, paper_keys):
  paper_keys_clear = {}
  for paper_i, keys in tqdm(paper_keys.items()):
    for key in keys:
      if key not in wiki_words_1k_frequency and len(key) not in [1,2,3]:
        if paper_i not in paper_keys_clear.keys():
          paper_keys_clear[paper_i] = set()
        paper_keys_clear[paper_i].add(key)
  return paper_keys_clear

#-----------------------------------------

# Rule: What is the racing division of Ferrari? => that is the racing division of Ferrari
def get_predicate(source):
    source_doc = stanza_nlp(source)
    sent = source_doc.sentences[0]
    if sent.words[-1].text == '?' and sent.words[0].xpos.startswith('W'):
        return 'the <mask> that ' + ' '.join([w.text for w in sent.words[1:-1]])
    return None

def getXlmRobertaTop1(xlm_roberta_base_output):
  """
  [{'score': 0.10563907772302628,
  'sequence': "Hello I'm a fashion model.",
  'token': 54543,
  'token_str': 'fashion'},...]
  """
  scores_list = [items['score'] for items in xlm_roberta_base_output]
  return '<' + xlm_roberta_base_output[scores_list.index(max(scores_list))]['token_str'] + '>'
    
  
def questionTurnDefination(qg_with_a_dict):
  """
  input: {paper_id: [[{"question":sentence,"answer":keyphrase}],
                    [{"question":sentence,"answer":keyphrase}]]}
  output: {keyphrase:set{definations}}
  """
  jargon_defination_dict = {}
  for _, qa_items in tqdm(qg_with_a_dict.items()):
    for [qa_item] in qa_items:
      #replace "what" to "the ____ that"
      defination_with_underline = get_predicate(qa_item["question"])
      if defination_with_underline == None:
        continue
      #---- use bert_fill_blank ----
      #missing_token = bert_fill_blank.predict(defination_with_underline)

      #---- use xlm-roberta-base ----
      missing_token = getXlmRobertaTop1(xlm_roberta_base_unmasker(defination_with_underline))

      defination_without_underline = defination_with_underline.replace('<mask>', missing_token)
      if qa_item["answer"] not in jargon_defination_dict.keys():
        jargon_defination_dict[qa_item["answer"]] = set()
      jargon_defination_dict[qa_item["answer"]].add(defination_without_underline)
  return jargon_defination_dict

#-----------------------------------------

#转换 qg_without_answer 格式
def toFormalKeys(qg_without_a_dict,keys_freq):
  keys_set = set(key for (key, n) in keys_freq)
  qg_without_a_dict_ = {}
  for paper_i, qa_items in tqdm(qg_without_a_dict.items()):
    for qa_item in qa_items:
     if qa_item["answer"] in keys_set:
       if paper_i not in qg_without_a_dict_:
           qg_without_a_dict_[paper_i] = []
       qg_without_a_dict_[paper_i].append([qa_item])
  return qg_without_a_dict_

#融合jargon_defination_dict_1, jargon_defination_dict_1
def mergerByDoc(jargon_defination_dict_1, jargon_defination_dict_2):
  """
  {keyphrase:set{definations}}
  """
  if len(set(jargon_defination_dict_1.keys())) != len(jargon_defination_dict_1.keys())\
  or len(set(jargon_defination_dict_2.keys())) != len(jargon_defination_dict_2.keys()):
    print("error!")
  jargon_defination_keys = set(jargon_defination_dict_1.keys()) | set(jargon_defination_dict_2.keys())
  print("number of jargon_defination length:{}".format(len(jargon_defination_keys)))
  jargon_defination_dict = {}
  for jargon in tqdm(jargon_defination_keys):
    if jargon in jargon_defination_dict_1.keys() and jargon in jargon_defination_dict_2.keys():
      jargon_defination_dict[jargon] = jargon_defination_dict_1[jargon] | jargon_defination_dict_2[jargon]
    elif jargon not in jargon_defination_dict_1.keys():
      jargon_defination_dict[jargon] = jargon_defination_dict_2[jargon]
    else:
      jargon_defination_dict[jargon] = jargon_defination_dict_1[jargon]
  return jargon_defination_dict

# Unsupervised Construction Layman Questions Dataset

def AG(questions, doc_list):
  ag_with_q_text_dict = {}
  for d_i, doc in tqdm(enumerate(doc_list)):
    for question in questions:
      QA_input = {'question': question, 'context': doc_list[d_i]}
      if len(QA_input) == 0: print("error of nothing")
      if d_i not in ag_with_q_text_dict.keys():
        ag_with_q_text_dict[d_i] = []
      ag_with_q_text_dict[d_i].append(nlp(QA_input))
  return ag_with_q_text_dict

def getRulesIndex(rules, answer_type):
  return [i for i, x in enumerate(rules) if x == answer_type]

def getQA(ag_with_q_dict,question_rules,title_list,rules, socre_aq_l,socre_aq_o):
  """
  output: QA  items['score'], items['answer'])
  """
  double_answer_type = ["languages","Issues","Fields","Methods","Achievements"]
  reverse_dictionary_qa_dict = []
  for paper_i, items in tqdm(ag_with_q_dict.items()):
    for answer_i, item in enumerate(items):
      if socre_aq_o > item['score'] > socre_aq_l:
        for question_rule in question_rules[answer_i].split(", "):
          question_with_underline, replace_types = question_rule.split(" | ")
          if len(replace_types.split()) == 1:
            question_replaced = question_with_underline.replace("****", item['answer'])
            reverse_dictionary_qa_dict.append({'question': question_replaced,
                                               'answer': title_list[paper_i]})
          else:
            question_replaced = question_with_underline.replace("****", item['answer'])
            if replace_types.split()[1] in double_answer_type:
              for answer_type_index in getRulesIndex(rules, replace_types.split()[1]):
                question_replaced = question_replaced.replace("----", items[answer_type_index]['answer'])
                reverse_dictionary_qa_dict.append({'question': question_replaced,
                                                   'answer': title_list[paper_i]})
  return reverse_dictionary_qa_dict


def getLaymanQA(ag_with_q_dict,question_rules,title_list, rules, jargon_defination_dict, score_ag_l,score_ag_o, score_cos):
  """
  output: QA  items['score'], items['answer'])
  """
  double_answer_type = ["languages","Issues","Fields","Methods","Achievements"]
  reverse_dictionary_qa_dict = []
  for paper_i, items in tqdm(ag_with_q_dict.items()):
    for answer_i, item in enumerate(items):
      # 判断 ：问答对的问题中的"术语"item['answer'] 是否在术语词典中
      if item['answer'] in jargon_defination_dict.keys():
        """
        {
          defination_type : set(jargon_defination,...)
        }
        """
        jargon_defination_dict_local = {}
        for jargon_defination_with_ in jargon_defination_dict[item['answer']]:
          jargon_defination = jargon_defination_with_.replace("<","").replace(">","")
          defination_type = jargon_defination_with_.split("<")[1].split(">")[0]

          if defination_type not in jargon_defination_dict_local.keys():
            jargon_defination_dict_local[defination_type] = set()
          jargon_defination_dict_local[defination_type].add(jargon_defination)

        if score_ag_o > item['score'] > score_ag_l:
          for question_rule in question_rules[answer_i].split(", "):
            question_with_underline, replace_types = question_rule.split(" | ")

            # rules[answer_i] vs jargon_defination_dict.keys()
            for defination_type in jargon_defination_dict_local.keys():
              if len(defination_type.lower()) == 0:
                #print("--",defination_type.lower(),"--",rules[answer_i])
                continue
              score = cosine_similarity([glove[defination_type.lower()]], [glove[rules[answer_i].lower()]])
              if score > score_cos:
                for defnition in jargon_defination_dict_local[defination_type]:
                  question_replaced = question_with_underline.replace("****", defnition)

                  if len(replace_types.split()) == 1:
                    reverse_dictionary_qa_dict.append({'question': question_replaced,
                                               'answer': title_list[paper_i]})
                  else:
                    if replace_types.split()[1] in double_answer_type:
                      for answer_type_index in getRulesIndex(rules, replace_types.split()[1]):
                        question_replaced = question_replaced.replace("----", items[answer_type_index]['answer'])
                        reverse_dictionary_qa_dict.append({'question': question_replaced,
                                                   'answer': title_list[paper_i]})
  return reverse_dictionary_qa_dict

def deduplication(reverse_dictionary_qa_dict):
  reverse_dictionary_qa_deduplication_dict = []
  deduplication_set = set()
  for item in reverse_dictionary_qa_dict:
    deduplication_set.add(item['answer'] + " ||| " + item['question'])
  for item in deduplication_set:
    reverse_dictionary_qa_deduplication_dict.append({'answer':item.split(" ||| ")[0],'question':item.split(" ||| ")[1]})
  print("length:{}".format(len(reverse_dictionary_qa_deduplication_dict)))
  return reverse_dictionary_qa_deduplication_dict


# Load xlm-roberta-base-squad2
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/xlm-roberta-base-squad2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def AGXlm(questions, doc_list):
  ag_with_q_text_dict = {}
  for d_i, doc in tqdm(enumerate(doc_list)):
    for question in questions:
      QA_input = {'question': question, 'context': doc_list[d_i]}
      if len(QA_input) == 0: print("error of nothing")
      if d_i not in ag_with_q_text_dict.keys():
        ag_with_q_text_dict[d_i] = []
      ag_with_q_text_dict[d_i].append(nlp(QA_input))
  return ag_with_q_text_dict



import re
import itertools

def clean_str(string,use=True):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    if not use: return string

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


# -------------------------- Train the paper embedding by TransE
def createTranseData(abstract_list, title_list):
  word2id = set()
  paper2id = set()
  relation_dict = dict()
  relation_triple = []

  for paper_i, abstract in tqdm(enumerate(abstract_list)):
    for word in clean_str(abstract).strip().split():
      word2id.add(word)
      paper2id.add(title_list[paper_i].replace(" ","_"))
      if word not in relation_dict.keys():
        relation_dict[word] = set()
      relation_dict[word].add(paper_i)
  
  for word_i, (_, titles) in tqdm(enumerate(relation_dict.items())):
    for tuple_2 in itertools.combinations(titles, 2):
      relation_triple.append(list(tuple_2).append(word_i))

  print("word2id:{} paper2id:{} relation_triple:{}".format(len(word2id),len(paper2id),len(relation_triple)))
  
  random.shuffle(relation_triple)
  return word2id, paper2id, relation_triple
