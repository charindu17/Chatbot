#implementation of a simple chatbot using tensorflow
import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer  #lemmatize words
from keras.models import load_model #load the pretained chatbot model


lemmatizer = WordNetLemmatizer()
intents = json.loads(open('C:\Users\Charindu Liyanage\Documents\GitHub\Chatbot\intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')#loading the pretrained model

#defining functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

#takes a sentence as input calls and clean up sentence to a list of lamitized words
#and create a bag of words to represent the sentence and convert it to numpy array
#bag of vector is a binary vector
def bag_of_words (sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class (sentence):
    bow = bag_of_words (sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes [r[0]], 'probability': str(r[1])})
    return return_list

#compare the responses and match it with the response of json file nad returns it as a chatbot response
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice (i['responses'])
            break
    return result

print("GO! Bot is running!")
 #infinite loop which waits for the user to input through the input fuction 
 #it calls the predict_class to predict intents of the message
 #get_response function retrieve the appropriate response from Json file
while True:
    message = input("")
    ints = predict_class (message)
    res = get_response (ints, intents)
    print (res)