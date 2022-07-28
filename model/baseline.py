import json
import pickle
import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class Baseline:
    """
    The class constructed to hold our baseline model logic.
    Same model architecture has been trained with data of size [10k, 50, 100k, 1m]
    """
    def __init__(self):
        self.models = None
        self.models_names = None
        self.max_seq_dict = None
        self.max_seq_list = None
        self.tokenizers = None

    def load(self, root):
        top10k = tensorflow.keras.models.load_model(f'{root}/top10k')
        top50k = tensorflow.keras.models.load_model(f'{root}/top50k')
        top100k = tensorflow.keras.models.load_model(f'{root}/top100k')
        top1m = tensorflow.keras.models.load_model(f'{root}/top1m')
        self.models = [top10k, top50k, top100k, top1m]

        self.models = list([
            ('top10k', top10k),
            ('top50k', top50k),
            ('top100k', top100k),
            ('top1m', top1m)
            # ('top5m', top5m) breaks the cloud
            # ('top10m', top10m) breaks the cloud
        ])
        self.models_names = [x[0] for x in self.models]
        with open('seq_dict.json', 'r') as f:
            self.max_seq_dict = json.load(f)
            self.max_seq_list = list(**self.max_seq_dict)

        for name, _ in self.models_names:
            self.tokenizers = []
            with open(f'{name}_tokenizer.pkl', 'rb') as f:
                self.tokenizers.append((name, pickle.load(f)))

        return self

    def tokenize_and_pad(self, df):
        """
        tokenizing and padding given df
        :param df: df assumed to have 'text' column
        :return: a tuple of all objects processed further on train
        """
        tokenizer = Tokenizer()
        corpus = df.loc[:, 'text']
        tokenizer.fit_on_texts(corpus)
        total_words = len(tokenizer.word_index) + 1

        # create input sequences using list of tokens
        input_sequences = []

        for line in corpus:
            token_list = tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i + 1]
                input_sequences.append(n_gram_sequence)

        # pad sequences
        max_sequence_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

        # create predictors and label
        predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
        label = ku.to_categorical(label, num_classes=total_words)

        return predictors, label, tokenizer, input_sequences, max_sequence_len

    def create_and_compile_model(self, tokenizer, input_sequences):
        """
        compiles model architecture matches current input.
        :param tokenizer: current model taokenizer
        :param input_sequences: models input
        :return:
        """
        total_words = len(tokenizer.word_index) + 1
        max_sequence_len = max([len(x) for x in input_sequences])
        model = Sequential()
        model.add(Embedding(total_words, 100, input_length=max_sequence_len - 1))
        model.add(Bidirectional(LSTM(150, return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(LSTM(100))
        model.add(Dense(total_words / 2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dense(total_words, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())

        return model

    def plot(self, history):
        """
        plots model performance
        :param history: history returned from model training
        """
        acc = history.history['accuracy']
        loss = history.history['loss']
        epochs = range(len(acc))
        plt.plot(epochs, acc, 'b', label='Training accuracy')
        plt.title('Training accuracy')
        plt.figure()
        plt.plot(epochs, loss, 'b', label='Training Loss')
        plt.title('Training loss')
        plt.legend()
        plt.show()

    def predict_text(self, model, tokenizer, max_sequence_len, seed_text="", next_words=20):
        """
        completes given sequence by maximal probability tokens
        :param model: current model
        :param tokenizer: current tokenizer
        :param max_sequence_len: the sequence of maximum length
        :param seed_text: text to start with
        :param next_words: amount of words to predict
        :return:
        """
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
            predicted = np.argmax(model.predict(token_list), axis=-1)
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            seed_text += " " + output_word
        return seed_text

    def save_model(self, model_name, model):
        """
        saves the model on root
        assumes the model is a tensforflow model
        :param model_name: model name
        :param model: actual model object
        """
        model.save(f'{model_name}')

    def save_tokenizer(self, tokenizer_name, tokenizer):
        """
        save the tokenizer on root
        assumes the tokenizer is tensorflow tokenizer
        :param tokenizer_name:
        :param tokenizer:
        :return:
        """
        with open(f'{tokenizer_name}_tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer, f)

    def train(self, dfs: list):
        """
        assumes a list of tuples that the first argument is the model name and the
        second is the dataframe that holds the data itself.
        :param dfs: list of tuples
        """
        self.max_seq_list = []
        for df in tqdm(dfs):
            predictors, labels, tokenizer, input_sequences, max_sequence_len = self.tokenize_and_pad(df[1])
            model = self.create_and_compile_model(tokenizer, input_sequences)
            history = model.fit(predictors, labels, epochs=300, verbose=0)
            self.plot(history)
            self.save_model(df[0], model)
            self.save_tokenizer(df[0], tokenizer)
            self.max_seq_list.append(max_sequence_len)


