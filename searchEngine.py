# import 
import re
import numpy as np
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer



def setup_connection():
    # connect to the local mongodb instance
    client = MongoClient('mongodb://localhost:27017/')
    # select the 'search_engine' database
    database = client['search_engine']
    return database


# normalize the text by converting to lowercase and removing non-alphanumeric characters
def normalize_text(input_text):
    # remove punctuation and non-word characters using regex and return lowercase version
    return re.sub(r'[^\w\s]', '', input_text).lower()


def generate_index(term_collection, doc_collection, documents):
    # insert each document into the 'documents' collection with a unique doc_id
    for idx, doc in enumerate(documents):
        doc_collection.insert_one({
            'doc_id': idx,  # unique identifier for each document
            'content': doc  # store the actual document text
        })

    # clean the documents: remove punctuation and convert to lowercase
    cleaned_docs = [normalize_text(doc) for doc in documents]

    # create a tf-idf vectorizer with n-grams from 1 to 3
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words=None)
    # fit the vectorizer on the cleaned documents and transform them into tf-idf matrix
    tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_docs)
    # extract vocabulary (mapping of terms to index positions in the matrix)
    vocab = tfidf_vectorizer.vocabulary_

    # populate the inverted index: for each term, store the documents where it appears
    for term_idx, (term, pos) in enumerate(vocab.items()):
        # get the tf-idf scores for each document for this term
        tfidf_scores = tfidf_matrix[:, pos].toarray().flatten()

        # list to store documents with non-zero tf-idf scores
        related_docs = []
        for doc_id in range(len(documents)):
            # check if the term appears in the document (tf-idf > 0)
            if tfidf_scores[doc_id] > 0:
                related_docs.append({
                    "doc_id": doc_id,  # reference the document id
                    "score": float(tfidf_scores[doc_id])  # store the tf-idf score for the term
                })

        # insert the term and associated document information into the 'terms' collection
        term_collection.insert_one({
            "_id": term_idx,  # unique id for the term
            "location": pos,  # position in the vocabulary (vector space)
            "docs_with_scores": related_docs  # list of documents with the tf-idf score for the term
        })

    # return the vectorizer and vocabulary for use in future searches
    return tfidf_vectorizer, vocab


def cosine_similarity(vec1, vec2):
    # calculate cosine similarity between two vectors
    # return 0 if either vector is all zeros
    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return 0.0

    # compute dot product of the vectors
    dot_product = np.dot(vec1, vec2)
    # compute the magnitudes (norms) of both vectors
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    # return cosine similarity: dot product divided by the product of magnitudes
    return dot_product / (norm_vec1 * norm_vec2) if norm_vec1 > 0 and norm_vec2 > 0 else 0.0


def perform_search(query, term_collection, doc_collection, vocab, vectorizer):
    # normalize the query text (remove punctuation, convert to lowercase)
    query_cleaned = normalize_text(query)
    # transform the query into a tf-idf vector using the same vectorizer
    query_vector = vectorizer.transform([query_cleaned]).toarray().flatten()

    # initialize a dictionary to store document vectors
    doc_vectors = {}
    # iterate over each term in the inverted index (terms collection)
    for term_entry in term_collection.find():
        # iterate over documents associated with this term
        for doc_info in term_entry["docs_with_scores"]:
            doc_id = doc_info["doc_id"]
            # if the document has not been encountered, initialize its vector
            if doc_id not in doc_vectors:
                doc_vectors[doc_id] = np.zeros(len(vocab))

            # assign the tf-idf score of the term to the document's vector at the appropriate position
            doc_vectors[doc_id][term_entry["location"]] = doc_info["score"]

    # list to store document scores based on cosine similarity
    ranked_docs = []
    for doc_id, doc_vector in doc_vectors.items():
        # calculate the similarity between the query vector and the document vector
        score = cosine_similarity(query_vector, doc_vector)
        # only add documents with non-zero similarity scores
        if score > 0:
            doc = doc_collection.find_one({"doc_id": doc_id})
            ranked_docs.append((doc["content"], score))

    # sort the documents in descending order of similarity score
    ranked_docs.sort(key=lambda x: x[1], reverse=True)
    # return the sorted list of documents and their similarity scores
    return ranked_docs


def main_execution():
    try:
        # set up the database connection
        db = setup_connection()
        # get references to the 'terms' and 'documents' collections
        terms_collection = db['terms']
        documents_collection = db['documents']

        # example set of documents
        docs = [
            "After the medication, headache and nausea were reported by the patient.",
            "The patient reported nausea and dizziness caused by the medication.",
            "Headache and dizziness are common effects of this medication.",
            "The medication caused a headache and nausea, but no dizziness was reported."
        ]

        # generate the inverted index and retrieve the vectorizer and vocabulary
        vectorizer, vocab = generate_index(terms_collection, documents_collection, docs)

        # example list of queries to search for
        queries_to_search = [
            "nausea and dizziness",
            "effects",
            "nausea was reported",
            "dizziness",
            "the medication"
        ]

        # iterate through each query and display the search results
        for i, query in enumerate(queries_to_search, 1):
            print(f"\nsearch query {i}: {query}")
            results = perform_search(query, terms_collection, documents_collection, vocab, vectorizer)
            # display the document content and the similarity score for each result
            for doc_content, score in results:
                print(f'"{doc_content}", score: {score:.2f}')

    except Exception as e:
        # handle any errors that occur during execution
        print("an error occurred:")
        print(e)


# execute the main function if the script is run directly
if __name__ == '__main__':
    main_execution()
