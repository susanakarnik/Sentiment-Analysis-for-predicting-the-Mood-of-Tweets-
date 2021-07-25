#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import pickle

from nltk.corpus import twitter_samples
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk import NaiveBayesClassifier
from nltk import classify


def clean_data(token):
    return [item for item in token if not item.startswith("http") and not item.startswith("@")]


def lemmatization(token):
    lemmatizer = WordNetLemmatizer()

    result = []
    for token, tag in pos_tag(token):
        tag = tag[0].lower()
        token = token.lower()
        if tag in "nva":
            result.append(lemmatizer.lemmatize(token, pos=tag))
        else:
            result.append(lemmatizer.lemmatize(token))
    return result


def remove_stop_words(token, stop_words):
    return [item for item in token if item not in stop_words]


def transform(token):
    result = {}
    for item in token:
        result[item] = True
    return result



def main():
    
    positive_tweets_tokens = twitter_samples.tokenized('positive_tweets.json')
    negative_tweets_tokens = twitter_samples.tokenized('negative_tweets.json')


    stop_words = stopwords.words('english')
    positive_tweets_tokens_cleaned = [remove_stop_words(lemmatization(clean_data(token)), stop_words) for token in positive_tweets_tokens]
    negative_tweets_tokens_cleaned = [remove_stop_words(lemmatization(clean_data(token)), stop_words) for token in negative_tweets_tokens]


    positive_tweets_tokens_transformed = [(transform(token), "Positive") for token in positive_tweets_tokens_cleaned]
    negative_tweets_tokens_transformed = [(transform(token), "Negative") for token in negative_tweets_tokens_cleaned]


    
    dataset = positive_tweets_tokens_transformed + negative_tweets_tokens_transformed
    random.shuffle(dataset)

    train_data = dataset[:7000]
    test_data = dataset[7000:]

    
    classifier = NaiveBayesClassifier.train(train_data)

    
    print("Accuracy is:", classify.accuracy(classifier, test_data))
    print(classifier.show_most_informative_features(10))

    
    f = open('my_classifier.pickle', 'wb')
    pickle.dump(classifier, f)
    f.close()


if __name__ == "__main__":
    main()

