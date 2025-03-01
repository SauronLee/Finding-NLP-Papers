<p align="center">
    <br>
    <img src="./logo_main.png" width="1000"/>
    <br>
<p>
<p align="center">
    <a href="https://colab.research.google.com/drive/1mbWcOZ5l97H5eSrFSH4dOUZVtsD5Sen4?usp=sharing">
        <img alt="colab" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>
    <a href="https://huggingface.co/SauronLee/BiLSTM_Finding_NLP_Papers">
        <img alt="inference" src="https://img.shields.io/badge/huggingface-inference-yellow">
    </a>
    <a>
        <img alt="build" src="https://img.shields.io/badge/build-passing-red">
    </a>
    <a>
        <img alt="license" src="https://img.shields.io/badge/license-MIT-green">
    </a>
</p>

This repository provides the source codes of "Finding NLP Papers by Asking a Multi-hop Question" published in Special Interest Group on Spoken Language Understanding and Dialogue Processing (SIG-SLUD) 2022

## Requirements

- tqdm
- transformers
- nltk
- dateparser
- scikit-learn
- fuzzywuzzy
- sentencepiece
- stanza

## Introduction
```python
import json
import os
import random
from tqdm import tqdm
import numpy as np
import argparse
from pretrained_models import T5_QG
import stanza
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from config import *
from pretrained_models import *
from utils import *
xlm_roberta_base_unmasker = pipeline('fill-mask', model='xlm-roberta-base')
nltk.download('punkt')
```
### Large Size Data

* [enwiki-20190320-words-frequency.txt (26.2MB)](https://drive.google.com/file/d/1juejElmlZ3HBI73A775WH9eZolPjgOIy/view?usp=sharing)
* [glove.6B.300d.txt (989.9MB)](https://drive.google.com/file/d/1Vc1pPVozq9eBD5NKDtCBZN-q7wpn-fzW/view?usp=sharing)
* [paper_data.npy (799.5MB)](https://drive.google.com/file/d/1QpLuhCHqbpnk2mNuKnXjhftXD0qul1ej/view?usp=sharing)
* [paper_qa_deduplication_dict_train_index.json (56.6MB)](https://drive.google.com/file/d/1-jKcrOY1QdJfjostuSDTNnKhSEQUoy26/view?usp=sharing)
* [paper_qa_deduplication_dict_train.json (84.7MB)](https://drive.google.com/file/d/1hrp_m-9mC--wTCNtoKiViySyb8oa055i/view?usp=sharing)
* [papers_embedding_inuse.json (85.1MB)](https://drive.google.com/file/d/103JS7_Nx0-2Gfhs77yp4g_a1Z8E4bjOw/view?usp=sharing)

### Create \{ jargon : defination }
<p align="center">
    <br>
    <img src="./defination_part.png" width="1000"/>
    <br>
<p>

### Building Schema
<p align="center">
    <br>
    <img src="./schema.png" width="1000"/>
    <br>
<p>

### Automatic Keyphrase Extraction

"ml6team/keyphrase-extraction-kbir-inspec"
```python
# Load pipeline
print("Load pipeline for ml6team/keyphrase-extraction-kbir-inspec")
model_name = "ml6team/keyphrase-extraction-kbir-inspec"
extractor = KeyphraseExtractionPipeline(model=model_name)

abstract = 'Sememes are defined as the atomic units to describe the semantic meaning of concepts. Due to the difficulty of manually annotating sememes and the inconsistency of annotations between experts, the lexical sememe prediction task has been proposed. However, previous methods heavily rely on word or character embeddings, and ignore the fine-grained information. In this paper, we propose a novel pre-training method which is designed to better incorporate the internal information of Chinese character. The Glyph enhanced Chinese Character representation (GCC) is used to assist sememe prediction. We experiment and evaluate our model on HowNet, which is a famous sememe knowledge base. The experimental results show that our method outperforms existing non-external information models.'

print(extractor(abstract))

output:
>> ['Glyph enhanced Chinese Character representation' 'HowNet'
>>  'character embeddings' 'lexical sememe prediction task'
>>  'sememe knowledge base']

print(qg_nlp.qg_without_answer(abstract))

output:
>> [{'answer': 'Sememes', 'question': 'What are defined as the atomic units to describe the semantic meaning of concepts?'}, 
>> {'answer': 'manually', 'question': 'How is annotating sememes difficult?'}, 
>> {'answer': 'word or character embeddings', 'question': 'What do previous methods heavily rely on?'}, 
>> {'answer': 'internal information of Chinese character', 'question': 'What is the Glyph enhanced Chinese Character representation (GCC) designed to better incorporate?'}, 
>> {'answer': 'Glyph enhanced Chinese Character representation', 'question': 'What is the GCC?'}, 
>> {'answer': 'HowNet', 'question': 'What is a famous sememe knowledge base?'}, 
>> {'answer': 'external information models', 'question': 'What does the Glyph enhanced Chinese Character representation outperform?'}]
```

### Convert a question into its declarative
```python
# Load xlm-roberta-base-squad2
model_name = "deepset/xlm-roberta-base-squad2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

getXlmRobertaTop1(xlm_roberta_base_unmasker("The <mask> that are defined as the atomic units to describe the semantic meaning of concepts ."))

output:
>> <terms>
```

## Train the paper embeddings by TransE
* install useing https://github.com/thunlp/OpenKE

## Train the paper embeddings by LINE
* install useing https://github.com/tangjianpku/LINE
    
# Citation
```   
@article{,
  title={Finding NLP Papers by Asking a Multi-hop Question},
  author={Li, Xiaoran and Takano, Toshiaki}
}
``` 
