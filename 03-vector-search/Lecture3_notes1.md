## Vector DB

### Why is it getting popular now?

- Reason #1: Over 80% of the data produced is unstructured. Example: Social Media, Image, Video, Audio
- Reason #2: LLM lacks long-term memory. Vector DBs provide ability to store and retrieve data for LLMs

### Vector Embeddings
- Used to transform words, sentences, and other data into numerical representations (vectors)
- They map different data types to points in a multidimensional space, with similar data points positioned near each other
- These numerical representations assist machines understand and process this data more effectively

### Types of Vector Embeddings

**Word Embeddings**
- Techniques: Word2Vec, GloVe, FastText
- Purpose: Capture semantic relationships and contextual information
**Sentence Embeddings**
- Models: Universal Sentence Encoder (USE), SkipThought
- Purpose: Represent overall meaning and context of sentences.
**Document Embeddings**
- Techniques: Doc2Vec, Paragraph Vectors
- Purpose: Capture semantic information and context of entire documents
**Image Embeddings**
- Techniques: CNN, ResNet, VGG
- Purpose: Capture visual features for tasks like classification and object detection


### Semantic Search with Elasticsearch

- run docker
```
docker run -it \
    --rm \
    --name elasticsearch \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:8.4.3
```

- Two very important concepts in Elasticsearch are documents and indexes
- A document is collection of fields with their associated values
- To work with Elasticsearch you have to organize your data into documents, and then add all your documents to an index
- Index is a collection of documents that is stored in a highly optimized format designed to perform efficient searches.

- Read the documents
```Python
with open('documents.json', 'rt') as f_in:
	docs_raw = json.load(f_in)

documents = []

for course_dict in docs_raw:
	for doc in course_dict['documents']:
		doc['course'] = course_dict['course']
		documents.append(doc)

documents[1]
```
- Step 2: Create Embeddings using Pretrained Models
```Python
from sentence_transformers import SentenceTransformer

# if you get an error do the following:
# 1. uninstall numpy
# 2. uninstall torch
# 3. pip install numpy==1.26.4
# 4. pip install torch
# run the above cell it should work

model = SentenceTransformer("all-mpnet-base-v2")

```

```Python
model.encode("this is a simple sentence")
```

```Python
operations = []
for doc in documents:
	doc["text_vector"] = model.encode(doc["text"]).tolist()
	operations.append(doc)
```
- Step 3: Setup Elasticsearch connection

```Python
from elasticsearch import Elasticsearch
es_client = Elasticsearch('http://localhost:9200')

es_client.info()
```

- Step 4: create mappings and index
	- Mapping is the process of defining how a document, and the fields it contains, are stored and indexed
	- Each document is a collection of fields, which each have their own data type.
	- We can compare mapping to a database schema in how it describes the fields and properties that documents hold, the datatype of each field (e.g., string, integer, or date), and how those fields should be indexed and stored.

```Python
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
            "course": {"type": "keyword"} ,
            "text_vector": {"type": "dense_vector", "dims": 768, "index": True, "similarity": "cosine"},
        }
    }
}
```
Note: you can get dims from ```len(model.encode("this is a simple sentence"))``` or from the model documentation. 
You can also choose different kinds of similarity metrics. Cosine is the most popular one.


```Python
index_name = "course-questions"

es_client.indices.delete(index=index_name, ignore_unavailable=True)
es_client.indices.create(index=index_name, body=index_settings)
```
Note: delete is just to make sure that you are actually updating the index, especially if you created one before. 


- Step 5: Add documents into index

```Python
for doc in operations:
    try:
        es_client.index(index=index_name, document=doc)
    except Exception as e:
        print(e)
```

- Step 6: Create end user query
```Python
search_term = "windows or mac?"
vector_search_term = model.encode(search_term)
```

```Python
query = {
    "field": "text_vector",
    "query_vector": vector_search_term,
    "k": 5,
    "num_candidates": 10000, 
}
```

```Python
res = es_client.search(index=index_name, knn=query, source=["text", "section", "question", "course"])
res["hits"]["hits"]
```

- Step 7: Perform Keyword search with Semantic Search (Hybrid/Advanced Search)
```Python
# Note: I made a minor modification to the query shown in the notebook here
# (compare to the one shown in the video)
# Included "knn" in the search query (to perform a semantic search) along with the filter  
knn_query = {
    "field": "text_vector",
    "query_vector": vector_search_term,
    "k": 5,
    "num_candidates": 10000
}
```

```Python
response = es_client.search(
    index=index_name,
    query={
        "match": {"section": "General course-related questions"},
    },
    knn=knn_query,
    size=5
)
```

```Python
response["hits"]["hits"]
```

