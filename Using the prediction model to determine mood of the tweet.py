#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pickle
import tweepy

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

def get_twitter_api():
    
    consumer_key = "uPRgRGfmgoPHJyuitn84v9hfk"
    consumer_secret = "IyjACTqz6Nn4QgaiBVhaqeTeFSZbbPa405CAb7526ZNddVjpUX"
    access_token = "949625398249144320-VUEtMbeltQ41giNp5er7IrXBvhtvB3t"
    access_token_secret = "rIJrXrhZKZaS6OAE5qLwaLPLRg0yUDyn554N8R8d57yog"

    
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

    
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    return api


def tokenize(tweet):
    return clean_data(tweet)



def get_classifier(pickle_name):
    f = open(pickle_name, 'rb')
    classifier = pickle.load(f)
    f.close()
    return classifier


def find_mood(search):
    classifier = get_classifier('my_classifier.pickle')

    api = get_twitter_api()

    stat = {
        "Positive": 0,
        "Negative": 0
    }
    for tweet in tweepy.Cursor(api.search, q=search).items(1000):
        custom_tokens = tokenize(tweet.text)

        category = classifier.classify(dict([token, True] for token in custom_tokens))
        stat[category] += 1

    print("The mood of", search)
    print(" - Positive", stat["Positive"], round(stat["Positive"]*100/(stat["Positive"] + stat["Negative"]), 1))
    print(" - Negative", stat["Negative"], round(stat["Negative"]*100/(stat["Positive"] + stat["Negative"]), 1))


if __name__ == "__main__":

    find_mood("#python")


# In[ ]:




