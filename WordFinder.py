import math
from textblob import TextBlob as tb

def tf(word, blob): #computes term frequency
    return blob.words.count(word) / len (blob.words)

def n_containing(word, bloblist): #returns the number of documents containing the word
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist): #computes inverse document frequency, more common word has a lower idf
    return math.log(len(bloblist) / (1+n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

document = tb("C:\Users\user1\PycharmProjects\deeplearning")

bloblist = [document]
for i, blob in enumerate(bloblist):
    print(format(i + 1))
    scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:3]:
        print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
