# Finding-NLP-Papers
## Introduction
### Automatic Keyphrase Extraction
"ml6team/keyphrase-extraction-kbir-inspec"
```python
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
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Load xlm-roberta-base
print('Loading xlm-roberta-base >>>>>>>>')
xlm_roberta_base_unmasker = pipeline('fill-mask', model='xlm-roberta-base')

# Load xlm-roberta-base-squad2
model_name = "deepset/xlm-roberta-base-squad2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

getXlmRobertaTop1(xlm_roberta_base_unmasker("The <mask> that are defined as the atomic units to describe the semantic meaning of concepts ."))

output:
>> <terms>
```

### 
 
```python

```
