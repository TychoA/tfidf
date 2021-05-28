# dependencies
from math import log10 as log

def term_frequency(term: str, document: dict):
    """
    Function to calculate the term frequency of a term in a document.
    """
    if term not in document:
        return 0

    number_of_occurences = document[term]
    total_number_of_occurences = sum(document.values())

    return number_of_occurences / total_number_of_occurences

def log_term_frequency(term: str, document: dict):
    """
    Function to calculate the log term frequency of a term in a document.
    """
    return 1 + log(term_frequency(term, document))

def document_frequency(term: str, documents: list):
    """
    Function to calculate the document frequency of a term in a collection of documents.
    """
    documents_with_term = [doc for doc in documents if term_frequency(term, doc) > 0];

    return len(documents_with_term)

def inverse_document_frequency(term: str, documents: list):
    """
    Function to calculate the inverse document frequency of a term in a collection of documents.
    """
    number_of_documents = len(documents)
    df = document_frequency(term, documents)

    return log(number_of_documents) - log(df)

def tfidf(term: str, document: dict, documents: list):
    """
    Function to calculate the term frequency inverse document frequency of a term
    in a document out of a collection of documents.
    """
    tf = term_frequency(term, document)
    idf = inverse_document_frequency(term, documents)

    return tf*idf

def tfidf_log(term: str, document: dict, documents: list):
    """
    Function to calculate the log term frequency inverse document frequency of a term
    in a document out of a collection of documents.
    """
    tf = log_term_frequency(term, document)
    idf = inverse_document_frequency(term, documents)

    return tf*idf

# test suite exception
class AssertError(Exception):

    def __init__(self, expected, actual):
        super().__init__(f"Assertion error: expected {expected}, got {actual} instead.")

def assert_equals(expected, actual):
    if abs(expected - actual) > 0.01:
        raise AssertError(expected, actual)
    return True

# Tests
if __name__ == "__main__":

    # setup test environment
    first_document = { "this": 1, "is": 1, "a": 2, "sample": 1 }
    second_document = { "this": 1, "is": 1, "another": 2, "example": 3 }
    documents = [first_document, second_document]

    # run test cases for "this"
    assert_equals(term_frequency("this", first_document), 0.2)
    assert_equals(term_frequency("this", second_document), 0.14)
    assert_equals(inverse_document_frequency("this", documents), 0)
    assert_equals(tfidf("this", first_document, documents), 0)
    assert_equals(tfidf("this", second_document, documents), 0)

    # run test cases for "example"
    assert_equals(term_frequency("example", first_document), 0)
    assert_equals(term_frequency("example", second_document), 0.429)
    assert_equals(inverse_document_frequency("example", documents), 0.301)
    assert_equals(tfidf("example", first_document, documents), 0)
    assert_equals(tfidf("example", second_document, documents), 0.129)
