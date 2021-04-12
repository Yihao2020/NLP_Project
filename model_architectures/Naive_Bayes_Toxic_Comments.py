import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

def main():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    train.head()

    test.head()

    print(train.shape)
    print()
    print(test.shape)
    print()


    train.info()
    print()

    test.info()
    print()

    features = ['toxic', 'severe_toxic', 'obscene', 'threat',
           'insult', 'identity_hate']

    length = len(features)

    for i in range(length):
        print(features[i])

    print()

    scores = []

    for i in range(length):
        print(features[i])
    # Create training and test sets
        X_train, X_test, y_train, y_test = train_test_split(train['comment_text'], train[features[i]], test_size=0.33, random_state=53)

    # Initialize a CountVectorizer object: count_vectorizer
        count_vectorizer = CountVectorizer(stop_words='english')

    # Transform the training data using only the 'text' column values: count_train
        count_train = count_vectorizer.fit_transform(X_train)

    # Transform the test data using only the 'text' column values: count_test
        count_test = count_vectorizer.transform(X_test)

        count_main_test = count_vectorizer.transform(test.comment_text)

    # Prints the first 10 features of the count_vectorizer
        #print(count_vectorizer.get_feature_names()[:10])

    # Initialize a TfidfVectorizer object: tfidf_vectorizer
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

    # Transform the training data: tfidf_train
        tfidf_train = tfidf_vectorizer.fit_transform(X_train)

    # Transform the test data: tfidf_test
        tfidf_test = tfidf_vectorizer.transform(X_test)

    # Prints the first 10 features
        print("First 10 Features:")
        print(tfidf_vectorizer.get_feature_names()[:10])

    # Instantiate a Multinomial Naive Bayes classifier: nb_classifier
        nb_classifier = MultinomialNB()

    # Fit the classifier to the training data
        nb_classifier.fit(count_train, y_train)

    # Create the predicted tags: pred
        pred = nb_classifier.predict(count_test)

    # Calculate the accuracy score: score
        score = metrics.accuracy_score(y_test, pred)
        scores.append(score)
        print("accuracy:")
        print(score)

    # Calculate the confusion matrix: cm
        cm = metrics.confusion_matrix(y_test, pred)
        print("confusion matrix:")
        print(cm)
        print()

    total = 0
    for s in scores:
        x = s*100
        total += x
    average_percent = total/len(scores)
    print("Average accuracy score:")
    print(average_percent)
    print()
    print("Done")

if __name__ == "__main__":
    main()
