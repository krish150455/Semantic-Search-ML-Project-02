#This project is jumping from exact keywords search to semantic search, i.e, searching based on meanings and intent.
#Vector Search is the particular mathematical mechanism which converts data into numerical embeddings and makes semantics possible.
#Vector search converts unstructured data (text, audio, images) into high-dimensional numerical arrays (vectors)
#and measures the mathematical distance (e.g., cosine similarity) between them.

#By TF-IDF, if we query feline, which is closely related to cats, output will be false due to keyword search.
#Embeddings create vectors based on same meaning rather than exact keywords...so feline gives right output

#TIP: For each result, check SHAPE(maybe (,384) or (1,384) and accordingly apply transformations)
#and TYPE, if NumPy, use reshape to change dims while if PyTorch use unsqueeze() and squeeze()

#Embedding Model
#Creates embeddings (vectors)

#Vector Search
#Compares vectors and retrieves nearest neighbors

#So:
#Sentence Transformer → creates embeddings
#Cosine Similarity / Vector DB → performs vector search

#sentence transformer used: https://www.sbert.net/?utm_source=chatgpt.com

from sentence_transformers import SentenceTransformer
import os

docs = {}
filenames = os.listdir("docs")  #returns all filenames
for filename in filenames:
    path = os.path.join("docs", filename)
    with open(path,"r") as file:
        docs[filename] = file.read()

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2") #Loading a pretrained sentence transformer model
embeddings = model.encode(list(docs.values()))
print(embeddings[0], embeddings.shape) #(4,384) means 4 documents with 384-dimensional embedding per document
print(type(embeddings))
cat_embedding = model.encode("cat")
feline_embedding = model.encode("feline")
cricket_embedding = model.encode("cricket")
print(cat_embedding.shape)

similarities = model.similarity(cat_embedding, feline_embedding)
print(similarities)
query="feline partner"
query_embedding = model.encode(query)
similarities = model.similarity(query_embedding, embeddings) #Instead of cosine similarity, model based similarity
print(similarities.shape)
scores = similarities.squeeze() #to reduce dimension
highest = scores.argmax() #givees index of highest value
print("Best Document:",list(docs.keys())[highest])
print("Similarity score:",scores[highest])
#VERY SIMILAR PATTERN TO FIRST PROJECT BUT INSTEAD OF VECTORIZER, WE'RE USING AN EMBEDDING MODEL.

