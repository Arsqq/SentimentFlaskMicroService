import json
import pathlib
import pandas as pd
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
import flask
import nltk
import spacy
from flask_cors import CORS
import jsonpickle
from flask import Flask, request, make_response, render_template
from flask_debugtoolbar import DebugToolbarExtension
from flask_eureka import Eureka
from py_eureka_client import eureka_client
import collections
from sentiment import get_entities_and_sentiment, get_tweets
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)
CORS(app)

port = 5000
eureka_client.init(eureka_server="http://localhost:8761/eureka",
                   app_name="flaskservice",
                   instance_port=port,
                   instance_ip="127.0.0.1",
                   instance_host="localhost",
                   )

result_list = []
tweets_list = []
text_data = ''


def get_occur(data):
    all_stopwords_gensim = STOPWORDS.union(
        {'#', 'http', '!', '@', ':', '.', '$', '?', '...', 'I', ',', '&', '', '*', 'New', 's', 'https', '0'})

    text_tokens = word_tokenize(data)
    tokens_without_sw = [word for word in text_tokens if not word in all_stopwords_gensim]
    most_occur = collections.Counter(map(str.lower, tokens_without_sw)).most_common(30)

    return most_occur


@app.route("/sentiment", methods=['POST'])
def analyse_text():
    content_list = []
    file = request.files.get('file')

    if ".txt" in file.filename:
        content_list = [line.decode('utf-8').strip() for line in file]
    if ".csv" in file.filename:
        df = pd.read_csv(file.stream)
        content_list = list(df['tweet_text'])

    for elem in content_list:
        sentiment, entities = get_entities_and_sentiment(elem)
        set_data = {"entities": entities, "sentiment": sentiment, "textContent": elem}
        result_list.append(set_data)
    response = flask.jsonify(result_list)
    response.headers.add('Access-Control-Allow-Origin', '*')
    text_file = request.files['file']
    text_file.seek(0)
    global text_data
    text_data = text_file.read()
    return response


@app.route("/sentimentList", methods=['GET'])
def get_result_list():
    response = flask.jsonify(result_list)
    response.headers.add('Access-Control-Allow-Origin', '*')
    result_list.clear()
    return response


@app.route("/words", methods=['GET'])
def get_words():
    counties = get_occur(text_data.decode('utf-8'))
    counties = json.dumps([{"text": ip[0], "value": ip[1]} for ip in counties])
    return counties


@app.route("/tweets", methods=['POST'])
def get_tweets_api():
    print(request.json['value'])
    data = request.json['value']
    tweets = get_tweets(data)

    for _ in tweets:
        sentiment, entities = get_entities_and_sentiment(_)
        set_data = {"entities": entities, "sentiment": sentiment}
        tweets_list.append(set_data)
    response = flask.jsonify(tweets_list)
    response.headers.add('Access-Control-Allow-Origin', '*')

    print(tweets_list)
    tweets_list.clear()
    return response


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=port)
