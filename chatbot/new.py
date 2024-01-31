#explanation in 41.00
import random
import json
import pickle
import numpy as np
import tensorflow as tf
#tensorflow is multidimensional array of elements
import nltk
from nltk.stem import WordNetLemmatizer
#initializing variables
lemmatizer = WordNetLemmatizer()

intents = json.loads(open('C:\Users\Charindu Liyanage\Documents\GitHub\Chatbot\intents.json').read())
#initializing variables
words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ','] #these will be neglected from a data set named JSON File

#add a tuple containg tuple list and an intent tag
for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#remove duplicates and show words in an alphabetical order
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))

classes = sorted(set(classes))

#pickling the sorted words and classes lists and save them
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []#initialize empty list
outputEmpty = [0] * len(classes)#creating a list of zero's

#tokenize each word in the document by limitizing it using wordnet 
for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

#returning the no of items in a container
trainX = training[:, :len(words)]
trainY = training[:, len(words):]

#linera stack
model = tf.keras.Sequential()

#designing the neural network model
#densely connected neural network
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))#adding a dropout layer to reduse it to 50% here
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

 #stochastic gradient descent optimizer with lerning rate of 0.01 and momentum 0f 0.9
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

#compiling the model to find the accuracy
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

 #traing data are sent to model as numpy arrays are saved in a variable
hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist) #save the train model to a file named chatbot history with the traing history
print('Done')#saved and excecuted successfully


