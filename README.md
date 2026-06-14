# Semantic-Search-ML-Project-02
# Project 02 - Semantic Search Engine using Sentence Transformers

## Overview

This project is the second step in my Retrieval-Augmented Generation (RAG) learning journey.

In Project 1, I built a keyword-based search engine using TF-IDF and cosine similarity. While effective for exact keyword matching, it struggled with semantic understanding. Queries containing synonyms or related concepts often failed because retrieval depended heavily on overlapping vocabulary.

To overcome this limitation, I built a semantic search engine using Sentence Transformers. Instead of matching exact words, documents and queries are converted into dense vector embeddings that capture meaning and context.

This enables retrieval based on semantic similarity rather than keyword overlap.

---

## Motivation

### Problem with TF-IDF

Consider the following example:

Query:

```text
feline pet
```

Document:

```text
Cats are small carnivorous mammals.
Many people keep cats as pets.
```

A TF-IDF based system may fail because the word **feline** never appears in the document.

TF-IDF primarily understands words.

```text
feline ≠ cats
```

---

### Solution: Embeddings

Sentence Transformers generate dense vector representations that capture semantic meaning.

```text
cat
↓
Embedding Vector

feline
↓
Embedding Vector
```

Since both words represent similar concepts, their vectors are located close together in embedding space.

This allows semantic retrieval even when exact keywords differ.

---

## Architecture

```text
Documents
    ↓
Sentence Transformer
    ↓
Document Embeddings
    ↓
User Query
    ↓
Query Embedding
    ↓
Similarity Search
    ↓
Most Relevant Document
```

---

## Technologies Used

* Python
* Sentence Transformers
* all-MiniLM-L6-v2
* NumPy

---

## Model Used

```python
SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)
```

This model converts text into 384-dimensional dense embeddings.

Example:

```text
4 Documents
↓
(4, 384)
```

Meaning:

* 4 document embeddings
* 384 features per embedding

---

## Implementation Details

### Document Loading

Text documents are loaded from a local folder and stored in a Python dictionary.

```python
{
    "france.txt": "...",
    "cats.txt": "...",
    "cricket.txt": "...",
    "space.txt": "..."
}
```

---

### Embedding Generation

Each document is converted into a dense vector embedding.

```python
embeddings = model.encode(
    list(docs.values())
)
```

Result:

```text
(4, 384)
```

---

### Semantic Similarity

The model is used to compare semantic similarity between concepts.

Example:

```python
cat_embedding
feline_embedding
```

Similarity scores demonstrate that semantically related words produce nearby vectors in embedding space.

---

### Query Retrieval

User queries are embedded using the same model.

```python
query_embedding = model.encode(query)
```

The query embedding is compared against all document embeddings.

The document with the highest similarity score is returned as the retrieval result.

---

## Example

### Query

```text
feline partner
```

### Retrieved Document

```text
cats.txt
```

Even though the word **feline** does not appear inside the document, the system successfully retrieves it because retrieval is based on meaning rather than exact keyword matches.

---

## Comparison with Project 01

| Feature              | Project 01 (TF-IDF) | Project 02 (Embeddings) |
| -------------------- | ------------------- | ----------------------- |
| Retrieval Type       | Keyword-Based       | Semantic                |
| Representation       | Sparse Vectors      | Dense Embeddings        |
| Understands Synonyms | No                  | Yes                     |
| Captures Meaning     | Limited             | Strong                  |
| Foundation for RAG   | Partial             | Yes                     |

---

## Key Takeaway

This project marks the transition from traditional information retrieval to modern AI-powered retrieval.

By replacing TF-IDF with semantic embeddings, the system can retrieve information based on meaning and intent rather than exact vocabulary.

This project forms the foundation for future work involving:

* Vector Databases
* ChromaDB
* Retrieval-Augmented Generation (RAG)
* Agentic RAG
* Multi-Hop Retrieval Systems
* Autonomous Research Agents
