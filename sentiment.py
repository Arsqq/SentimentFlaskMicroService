import collections
import string
from typing import Tuple, List

from flair.data import Sentence
from flair.models import TextClassifier

import spacy
import snscrape.modules.twitter as sntwitter

nlp = spacy.load("en_core_web_trf")
sentiment_model = TextClassifier.load('sentiment-fast')


def get_entities_and_sentiment(text: str) -> Tuple[dict, List[dict]]:
    """Parse a string, and determine sentiment polarity and entities contained within"""
    doc = nlp(text)
    entity_list = [
        {"name": x.text, "type": x.label_} for x in doc.ents
    ]
    sentence = Sentence(text)
    sentiment_model.predict(sentence)
    label = sentence.labels[0]
    if label.score < 0.65:
        label.value = "NEUTRAL"
    sentiment = {'sentiment': label.value, 'polarity': label.score}
    return sentiment, entity_list


def get_tweets(query: str):
    tweets = []
    for i, tweet in enumerate(
            sntwitter.TwitterSearchScraper(query).get_items()):
        if i > 2000:
            break
        tweets.append(tweet.content)
    return tweets
