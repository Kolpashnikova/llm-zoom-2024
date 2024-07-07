- Why do we need evaluation
- [Evaluation metrics](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/03-vector-search/evaluation-metrics.md)
	- using different methods/embeddings/boosting depends on the data, etc. basically depends case by case, so we need evalutaions.
	- in this course, we will use hit rate and MRR, but there are multiple different metrics. These just fit out purpose.
- Ground truth / gold standard data
	- For each record in FAQ:
		- generate 5 questions
	- in our dataset, we have 1000, so we will have 5000 query results.
	- And for this, we will use LLMs.
```Python
import requests

docs_uri = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
	course_name = course['course']

	for doc in course['documents']:
		doc['course'] = course_name
		documents.append(doc)
```

```Python
import hashlib

def generate_document_id(doc):
	combined = f"{doc['course']}-{doc['question']}-{doc['text'][:10]}"
	hash_object = hashlib.md5(combined.encode())
	hash_hex = hash_object.hexdigest()
	document_id = hash_hex[:8]
	return document_id
```

```Python
for doc in documents:
	doc['id'] = generate_document_id(doc)
```

To check if our hashing system is enoughly unique (so len of combined and variety is ok), you can check the following:
```Python
from collections import defaultdict

hashes = defaultdict(list)
for doc in documents:
	doc_id = doc['id']
	hashes[doc_id].append(doc)

len(hashes), len(documents) # should be the same if hashing is ok aka hash collision
```

```Python
with open('documents-with-ids.json', 'wt') as f_out:
	json.dump(documents, f_out, indent = 2)

```

- Generating ground truth with LLM
```Python
prompt_template = """
You emulate a student taking our course. 
Formulate 5 questions this student might ask based on a FAQ record. The record should contain the answer to the questions, and the questions should be complete and not too short. 
If possible, use as fewer words as possible from the record.

The record:

section: {section}
question: {question}
answer: {text}

Provide the output in parsable JSON without using code blocks:

["question1", "question2", ..., "question5"]
""".strip()

```

```Python
from openai import OpenAI
client = OpenAI()
```

```Python
doc = documents[2]
prompt = prompt_template.format(**doc)
```

```Python
def generate_questions(doc):
	prompt = prompt_template.format(**doc)

	reponse = client.chat.completions.create(
		model='gpt-4o'
		messages=[{"role": "user", "content": prompt}]
	
	)

	json_reponse = response.choices[0].message.content
	return json_response
```

```Python
json_response = generate_questions(doc)

json.loads(json_response)
```

```Python
from tqdm.auto import tqdm

results = []

for doc in tqdm(documents):
	doc_id = doc['id']
	if doc_id in result:
		continue
	questions = generate_questions(doc)
	results[doc_id] = questions

```

```Python
import pickle

with open('results.bin', 'rb') as f_in:
	results = pickle.load(f_in)
```

```Python
parsed_results = []

for doc_id, json_questions in results.items():
	parsed_results[doc_id] = json.loads(json_questions)
```

```Python
doc_index = {d['id']: d for d in documents}

final_results = []

for doc_id, questions in parsed_results.items():
	course = doc_index[doc_id]['course']
	for q in questions:
		final_results.append((q, course, doc_id))

import pandas as pd

df = pd.DataFrame(final_results, columns = ['question', 'course', 'document'])

df.to_csv('ground_truth_dat.csv', index = False)
```

- Evaluating the search result


```Python
import json

with open('documents-with-ids.json', 'rt') as f_in:
    documents = json.load(f_in)
```

```Python
from elasticsearch import Elasticsearch

es_client = Elasticsearch('http://localhost:9200') 

index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"},
            "id": {"type": "keyword"}, # this field is new
        }
    }
}

index_name = "course-questions"

es_client.indices.delete(index=index_name, ignore_unavailable=True)
es_client.indices.create(index=index_name, body=index_settings)
```

```Python
from tqdm.auto import tqdm

for doc in tqdm(documents):
    es_client.index(index=index_name, document=doc)

```

```Python
def elastic_search(query, course):
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": course
                    }
                }
            }
        }
    }

    response = es_client.search(index=index_name, body=search_query)
    
    result_docs = []
    
    for hit in response['hits']['hits']:
        result_docs.append(hit['_source'])
    
    return result_docs
```

```Python
elastic_search(
    query="I just discovered the course. Can I still join?",
    course="data-engineering-zoomcamp"
)
```

```Python
import pandas as pd

df_ground_truth = pd.read_csv('ground-truth-data.csv')
ground_truth = df_ground_truth.to_dict(orient='records')

relevance_total = []

for q in tqdm(ground_truth):
    doc_id = q['document']
    results = elastic_search(query=q['question'], course=q['course'])
    relevance = [d['id'] == doc_id for d in results]
    relevance_total.append(relevance)

example = [
    [True, False, False, False, False], # 1 (hit rate)
    [False, False, False, False, False], # 0
    [False, False, False, False, False], # 0 
    [False, False, False, False, False], # 0
    [False, False, False, False, False], # 0 
    [True, False, False, False, False], # 1
    [True, False, False, False, False], # 1
    [True, False, False, False, False], # 1
    [True, False, False, False, False], # 1
    [True, False, False, False, False], # 1 
    [False, False, True, False, False],  # 1
    [False, False, False, False, False], # 0
]

# for MRR:
# 1 => 1
# 2 => 1 / 2 = 0.5
# 3 => 1 / 3 = 0.3333
# 4 => 0.25
# 5 => 0.2
# rank => 1 / rank
# none => 0
```
	- hit-rate(recall)
	- mean reciprocal rank (mrr)


```Python
def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)

def mrr(relevance_total):
    total_score = 0.0

    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relevance_total)

hit_rate(example)
```


```Python
mrr(example)
```

```Python
hit_rate(relevance_total), mrr(relevance_total)
```

```Python
import minsearch

index = minsearch.Index(
    text_fields=["question", "text", "section"],
    keyword_fields=["course", "id"]
)

index.fit(documents)
```

```Python
def minsearch_search(query, course):
    boost = {'question': 3.0, 'section': 0.5}

    results = index.search(
        query=query,
        filter_dict={'course': course},
        boost_dict=boost,
        num_results=5
    )

    return results

relevance_total = []

for q in tqdm(ground_truth):
    doc_id = q['document']
    results = minsearch_search(query=q['question'], course=q['course'])
    relevance = [d['id'] == doc_id for d in results]
    relevance_total.append(relevance)
```

```Python
hit_rate(relevance_total), mrr(relevance_total)
```


Just generalizing the function for whichever function is used for search:
```Python
def evaluate(ground_truth, search_function):
    relevance_total = []

    for q in tqdm(ground_truth):
        doc_id = q['document']
        results = search_function(q)
        relevance = [d['id'] == doc_id for d in results]
        relevance_total.append(relevance)

    return {
        'hit_rate': hit_rate(relevance_total),
        'mrr': mrr(relevance_total),
    }
```

```Python
evaluate(ground_truth, lambda q: elastic_search(q['question'], q['course']))
```

```Python
evaluate(ground_truth, lambda q: minsearch_search(q['question'], q['course']))
```
### Evaluating Retrieval

```Python
import json

with open('documents-with-ids.json', 'rt') as f_in:
    documents = json.load(f_in)
```

```Python
from sentence_transformers import SentenceTransformer

model_name = 'multi-qa-MiniLM-L6-cos-v1'
model = SentenceTransformer(model_name)

v = model.encode('I just discovered the course. Can I still join?')

len(v)
```

```Python
from elasticsearch import Elasticsearch

es_client = Elasticsearch('http://localhost:9200') 

index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"},
            "id": {"type": "keyword"},
            "question_vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            },
            "text_vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            },
            "question_text_vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            },
        }
    }
}

index_name = "course-questions"

es_client.indices.delete(index=index_name, ignore_unavailable=True)
es_client.indices.create(index=index_name, body=index_settings)
```

```Python
from tqdm.auto import tqdm
for doc in tqdm(documents):
    question = doc['question']
    text = doc['text']
    qt = question + ' ' + text

    doc['question_vector'] = model.encode(question)
    doc['text_vector'] = model.encode(text)
    doc['question_text_vector'] = model.encode(qt)
```

```Python
for doc in tqdm(documents):
    es_client.index(index=index_name, document=doc)
```

```Python
query = 'I just discovered the course. Can I still join it?'

v_q = model.encode(query)

def elastic_search_knn(field, vector, course):
    knn = {
        "field": field,
        "query_vector": vector,
        "k": 5,
        "num_candidates": 10000,
        "filter": {
            "term": {
                "course": course
            }
        }
    }

    search_query = {
        "knn": knn,
        "_source": ["text", "section", "question", "course", "id"]
    }

    es_results = es_client.search(
        index=index_name,
        body=search_query
    )
    
    result_docs = []
    
    for hit in es_results['hits']['hits']:
        result_docs.append(hit['_source'])

    return result_docs

def question_vector_knn(q):
    question = q['question']
    course = q['course']

    v_q = model.encode(question)

    return elastic_search_knn('question_vector', v_q, course)
```

```Python
import pandas as pd
df_ground_truth = pd.read_csv('ground-truth-data.csv')

ground_truth = df_ground_truth.to_dict(orient='records')

def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)

def mrr(relevance_total):
    total_score = 0.0

    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relevance_total)

def evaluate(ground_truth, search_function):
    relevance_total = []

    for q in tqdm(ground_truth):
        doc_id = q['document']
        results = search_function(q)
        relevance = [d['id'] == doc_id for d in results]
        relevance_total.append(relevance)

    return {
        'hit_rate': hit_rate(relevance_total),
        'mrr': mrr(relevance_total),
    }

evaluate(ground_truth, question_vector_knn)
```

```Python
def text_vector_knn(q):
    question = q['question']
    course = q['course']

    v_q = model.encode(question)

    return elastic_search_knn('text_vector', v_q, course)

evaluate(ground_truth, text_vector_knn)

```

```Python

def question_text_vector_knn(q):
    question = q['question']
    course = q['course']

    v_q = model.encode(question)

    return elastic_search_knn('question_text_vector', v_q, course)

evaluate(ground_truth, question_text_vector_knn)
```

```Python
def elastic_search_knn_combined(vector, course):
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": [
                    {
                        "script_score": {
                            "query": {
                                "term": {
                                    "course": course
                                }
                            },
                            "script": {
                                "source": """
                                    cosineSimilarity(params.query_vector, 'question_vector') + 
                                    cosineSimilarity(params.query_vector, 'text_vector') + 
                                    cosineSimilarity(params.query_vector, 'question_text_vector') + 
                                    1
                                """,
                                "params": {
                                    "query_vector": vector
                                }
                            }
                        }
                    }
                ],
                "filter": {
                    "term": {
                        "course": course
                    }
                }
            }
        },
        "_source": ["text", "section", "question", "course", "id"]
    }

    es_results = es_client.search(
        index=index_name,
        body=search_query
    )
    
    result_docs = []
    
    for hit in es_results['hits']['hits']:
        result_docs.append(hit['_source'])

    return result_docs
```

```Python
def vector_combined_knn(q):
    question = q['question']
    course = q['course']

    v_q = model.encode(question)

    return elastic_search_knn_combined(v_q, course)

evaluate(ground_truth, vector_combined_knn)
```