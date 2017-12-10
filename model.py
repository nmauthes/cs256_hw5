'''

A recurrent neural network with LSTM for generating music via text (.abc) files

'''

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import SGD

import numpy as np
import random
import os
import argparse

from abc_utils import generate_data_file


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
        model.add(Dropout(0.2))
        model.add(Dense(len(chars)))
        model.add(Activation('softmax'))

    optimizer = SGD(lr=0.1, momentum=0.6)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    ###################################################################

    if(args.mode == 'train'):
        print('Training...')
    else:
        print('Generating...')

    out_name = f'{args.model_name}_{args.diversity}_{args.mode}.abc'
    generate = True if args.mode == 'generate' or args.generate_while_training else False

    if generate:
        # Save the output to file
        if not os.path.exists(args.out_folder):
            try:
                os.makedirs(args.out_folder)
            except OSError:
                raise

        out_file = open(os.path.join(args.out_folder, out_name), 'w')

    # train the model, output generated text after each iteration
    for iteration in range(args.num_iterations):
        print(f'Iteration {iteration + 1}/{args.num_iterations}')

        if args.mode == 'train':
            model.fit(x, y,
                      batch_size=128,
                      epochs=1)

        if generate:
            start_index = random.randint(0, len(text) - maxlen - 1)

            generated = ''
            sentence = text[start_index: start_index + maxlen] # random seed
            #sentence = text[0: maxlen] # seed with first maxlen chars
            generated += sentence

            for i in range(500):
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.

                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, args.diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                out_file.write(next_char)

    if generate:
        print(f'Saved file as {out_name}')
        out_file.close()

    if args.mode == 'train': # save model
        model.save(args.model_name)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# Command-line args
parser = argparse.ArgumentParser(
    description='RNN implementation using LSTM for generating music via text (.abc) files',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='For further questions, please consult the README.'
)

parser.add_argument(
    'diversity',
    type=float,
    help='The temperature of the model (degree of randomness in generated text), typically a float from 0 to 1'
)
parser.add_argument(
    'num_iterations',
    type=int,
    help='If mode=\'train\' this is number of epochs to train. If mode=\'generate\' this is number of times to generate'
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
parser.add_argument(
    '--out_folder',
    default='output',
    help='The path to the folder where output will be stored. Default is \'output\'.'
)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
