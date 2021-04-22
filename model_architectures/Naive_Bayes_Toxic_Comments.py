import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

def main():
    train = pd.read_csv("../data/train.csv")

    train.head()


    print(train.shape)
    print()


    train.info()
    print()


    features = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    length = len(features)

    for i in range(length):
        print(features[i])

    print()

    scores = []
    recalls = []
    precisions = []
    F1s = []

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
    # Recall
        recall = metrics.recall_score(y_test, pred)
        recalls.append(recall)
        print("Recall Score:")
        print(recall)
        print()
    #Precision
        precision = metrics.precision_score(y_test, pred)
        precisions.append(precision)
        print("Precision Score:")
        print(precision)
        print()
    #F1
        F1 = metrics.f1_score(y_test, pred)
        F1s.append(F1)
        print("f1 Score:")
        print(F1)
        print()
    #Final
        print(metrics.classification_report(y_test, pred))
        print()

    total = 0
    for s in scores:
        total += s

    average_percent = total/len(scores)
    print("Average Accuracy Score:")
    print(average_percent)
    print()

    total_recall = 0
    for r in recalls:
        total_recall += r

    average_recall = total_recall / len(recalls)
    print("Average Recall Score:")
    print(average_recall)
    print()

    total_pre = 0
    for p in precisions:
        total_pre += p

    average_pre = total_pre / len(precisions)
    print("Average Precision Score:")
    print(average_pre)
    print()

    total_f1 = 0
    for f in F1s:
        total_f1 += f

    average_f1 = total_f1 / len(F1s)
    print("Average F1 Score:")
    print(average_f1)
    print()

    print("Done")

if __name__ == "__main__":
    main()
