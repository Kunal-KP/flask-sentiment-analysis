import pickle
from sklearn.feature_extraction.text import CountVectorizer
from clean_reviews import review_cleaner

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

X_test = "enjoyed it a lot."
#X_test = "good moview liked enjoyed lot"
cleaned_rev = review_cleaner(X_test)
print('Cleaned review: ',cleaned_rev)

test_bag = vectorizer.transform([X_test]).toarray()

prediction = model.predict(test_bag)

print('Prediction: ',prediction)

