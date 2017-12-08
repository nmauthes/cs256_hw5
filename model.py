'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop

import numpy as np
import random
import sys
import os
import argparse

from abc_utils import generate_data_file


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def main(args):
    # Load the data
    generate_data_file(args.training_folder, args.training_file)
    text = open(args.training_file).read()

    chars = sorted(list(set(text)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = 40
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])

    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    # Load the model, or build a new one
    if os.path.isfile(args.model_name):
        model = load_model(args.model_name)
    elif args.mode == 'generate':
        raise Exception('MODEL FILE NOT FOUND!')
    else:
        model = Sequential()
        model.add(LSTM(128, input_shape=(maxlen, len(chars))))
        model.add(Dense(len(chars)))
        model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    ###################################################################

    # train the model, output generated text after each iteration
    for iteration in range(args.num_epochs):
        if args.mode == 'train':
            print()
            print('-' * 50)
            print('Iteration', iteration)
            model.fit(x, y,
                      batch_size=128,
                      epochs=1)

        if args.mode == 'generate' or args.generate_while_training:
            start_index = random.randint(0, len(text) - maxlen - 1)

            print()
            print('----- diversity:', args.diversity)

            generated = ''
            sentence = text[start_index: start_index + maxlen]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for i in range(400):
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.

                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, args.diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()

            if args.mode == 'generate':
                break

    if args.mode == 'train': # save model
        model.save(args.model_name)


# Command-line args
parser = argparse.ArgumentParser(
    description='RNN implementation using LSTM for generating music via text (.abc) files',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='For further questions, please consult the README.'
)

parser.add_argument(
    'diversity',
    type=float,
    help='The temperature of the model (degree of randomness in generated text) [0.0, inf)'
)
parser.add_argument(
    'num_epochs',
    type=int,
    help='The number of epochs (iterations) to run the model [0, inf)'
)
parser.add_argument(
    'mode',
    choices=['train', 'generate'],
    help='Train a new model or use a saved model to generate [train | generate]'
)
parser.add_argument(
    'model_name',
    help='''The name of the saved model to use, if available. If mode=\'train\' and the model isn\'t found it will
    create a new one with name model_name.'''
)

# Optional args
parser.add_argument(
    '--training_folder',
    default='nottingham_database',
    help='The path to the folder containing .abc files. Default is nottingham_database folder included with project.'
)
parser.add_argument(
    '--training_file',
    default='data.txt',
    help='The name of the .txt consisting of concatenated .abc\'s that will be used for training. Default \'data.txt\'.'
)
parser.add_argument(
    '--generate_while_training',
    action='store_true',
    default=False,
    help='Whether to generate output after each epoch during training. Default is True.'
)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)