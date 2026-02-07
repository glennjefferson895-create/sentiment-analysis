import pandas as pd
import nltk 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('movie_reviews')

documents = [
    (" ".join(nltk.corpus.movie_reviews.words(fileid)), category)
    for category in nltk.corpus.movie_reviews.categories()
    for fileid in nltk.corpus.movie_reviews.fileids(category)
]

myTable = pd.DataFrame(documents, columns=["review", "sentiment"])

myVectorizer = CountVectorizer()
X = myVectorizer.fit_transform(myTable["review"])
y = myTable["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.2, stratify = y)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"The accuracy of the model: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
print(nltk.data.path)
