# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import imp
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

MAX_LENGTH = 15
save_dir = "./save"

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3

class Voc:
    def __init__(self, name):
        self.name = name
        # self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2:"EOS", 3:"UNK"}
        self.n_words = 4  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, n=10000):
        words = sorted(self.word2count.keys(), key=lambda x: self.word2count[x], reverse=True)
        words = words[:(n - 4)]

        # words_left = words_a[(n - 4):len(words_a)]
        # count_choose = 0
        # for word in words:
        #     count_choose += self.word2count[word]

        # count_not_choose = 0
        # for word in words_left:
        #     count_not_choose += self.word2count[word]
        # print(count_choose)
        # print(count_not_choose)

        self.index2word = {0: "PAD", 1: "SOS", 2:"EOS", 3:"UNK"}
        self.word2index = {"PAD": 0, "SOS": 1, "EOS": 2, "UNK": 3}
        self.n_words = 4

        # count = 0
        for word in words:
            self.index2word[self.n_words] = word
            self.word2index[word] = self.n_words
            self.n_words += 1

            # if count < 100 :
            #     print(word)
            #     print(self.word2count[word])
            #     count += 1
    # def trim_temp(self, min_count):
    #     if self.trimmed:
    #         return
    #     self.trimmed = True

    #     keep_words = []

    #     for k, v in self.word2count.items():
    #         if v >= min_count:
    #             keep_words.append(k)

    #     print('keep_words {} / {} = {:.4f}'.format(
    #         len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
    #     ))

    #     # Reinitialize dictionaries
    #     self.word2index = {"PAD": 0, "SOS": 1, "EOS": 2, "UNK": 3}
    #     self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token:"UNK"}
    #     self.n_words = 4 # Count default tokens

    #     for word in keep_words:
    #         self.index2word[self.n_words] = word
    #         self.word2index[word] = self.n_words
    #         self.n_words += 1

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    # s = unicodeToAscii(s.lower().strip())
    s = unicodeToAscii(s.strip())
    s = re.sub(r"([!?])", r" \1", s)
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def inputNormalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def readVocs(sequence_input, sequence_output):
    print("Reading lines...")

    # combine every two lines into pairs and normalize
    # with open(corpus) as f:
    #     content = f.readlines()
    # import gzip
    # content = gzip.open(corpus, 'rt')
    # lines = [x.strip() for x in content]
    # it = iter(lines)
    # # pairs = [[normalizeString(x), normalizeString(next(it))] for x in it]
    # pairs = [[x, next(it)] for x in it]

    api_sequence = open('api_sequence/%s.csv' % sequence_input, encoding = 'utf-8').read().strip().split('\n')
    api_usage = open('api_usage/%s.csv' % sequence_output, encoding = 'utf-8').read().strip().split('\n')

    print(len(api_sequence))

    pairs = list(zip([inputNormalizeString(sequence) for sequence in api_sequence], [normalizeString(usage) for usage in api_usage]))
    # lines = open('data/%s-%s.txt' % (sequence_output, sequence_input), encoding='utf-8').read().strip().split('\n')

    # # Split every line into pairs and normalize
    # pairs = [[inputNormalizeString(s) for s in l.split('\t')] for l in lines]

    # pairs = [list(reversed(p)) for p in pairs]

    input_sequence = Voc(sequence_input)
    output_sequence = Voc(sequence_output)
    return input_sequence, output_sequence, pairs

def filterPair(p):
    # input sequences need to preserve the last word for EOS_token
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(sequence_input, sequence_output, corpus):
    input_sequence, output_sequence, pairs = readVocs(sequence_input, sequence_output)
    print("Read %s sentense pairs" % len(pairs))
    pairs = filterPairs(pairs) #裁剪数据集（保留较为简单的语句进行训练，充分训练应该注释该段语句）
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_sequence.addSentence(pair[0])
        output_sequence.addSentence(pair[1])
    print("Counted words: (before)")
    print(input_sequence.name, input_sequence.n_words)
    print(output_sequence.name, output_sequence.n_words)

    print("update the words")
    input_sequence.trim(n=10000)
    output_sequence.trim(n=10000)
    # pairs = trimRareWords(input_sequence, output_sequence, pairs, 5)

    print("Counted words: (after)")
    print(input_sequence.name, input_sequence.n_words)
    print(output_sequence.name, output_sequence.n_words)

    directory = os.path.join(save_dir, 'training_data', corpus)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(input_sequence, os.path.join(directory, '{!s}.tar'.format('input_sequence')))
    torch.save(output_sequence, os.path.join(directory, '{!s}.tar'.format('output_sequence')))
    torch.save(pairs, os.path.join(directory, '{!s}.tar'.format('pairs')))

    return input_sequence, output_sequence, pairs

def loadPrepareData(corpus):
    # corpus_name = corpus.split('/')[-1].split('.')[0]
    # try:
    #     print("Start loading training data ...")
    #     voc = torch.load(os.path.join(save_dir, 'training_data', corpus_name, 'voc.tar'))
    #     pairs = torch.load(os.path.join(save_dir, 'training_data', corpus_name, 'pairs.tar'))
    # except FileNotFoundError:
    #     print("Saved data not found, start preparing trianing data ...")
    #     voc, pairs = prepareData(corpus, corpus_name)
    # return voc, pairs

    try:
        print("Start loading training data ...")
        input_sequence = torch.load(os.path.join(save_dir, 'training_data', corpus, 'input_sequence.tar'))
        output_sequence = torch.load(os.path.join(save_dir, 'training_data', corpus, 'output_sequence.tar'))
        pairs = torch.load(os.path.join(save_dir, 'training_data', corpus, 'pairs.tar'))
    except FileNotFoundError:
        input_sequence, output_sequence, pairs = prepareData('api_sequence', 'api_usage', corpus)
        # input_sequence, output_sequence, pairs = prepareData('fra', 'eng', corpus)
        print(random.choice(pairs))
    
    return input_sequence, output_sequence, pairs

corpus_name = "data_model"

input_sequence, output_sequence, pairs = loadPrepareData(corpus_name)
# Print some pairs to validate
print("\npairs:")
for pair in pairs[:10]:
    print(pair)


def indexesFromSentence(voc, sentence):
    return [voc.word2index.get(word, UNK_token) for word in sentence.split(' ')] + [EOS_token]

def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Returns all items for a given batch of pairs
def batch2TrainData(input_sequence, output_sequence, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, input_sequence)
    output, mask, max_target_len = outputVar(output_batch, output_sequence)
    return inp, lengths, output, mask, max_target_len

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden


# Luong attention layer
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden

class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores

def evaluate(encoder, decoder, searcher, input_sequence, output_sequence, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(input_sequence, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [output_sequence.index2word[token.item()] for token in tokens]
    return decoded_words
            

model_name = 'data_model'
# attn_model = 'dot'
attn_model = 'general'
#attn_model = 'concat'
hidden_size = 256
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
checkpoint_iter = 4000
# loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))

# Load model if a loadFilename is provided
loadFilename = os.path.join("checkpoint.tar")
if loadFilename:
    # If loading on same machine the model was trained on
    print("loading the model")
    #checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']

print('Building encoder and decoder ...')
# Initialize word embeddings
input_embedding = nn.Embedding(input_sequence.n_words, hidden_size)

# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, input_embedding, encoder_n_layers, dropout)
output_embedding = nn.Embedding(output_sequence.n_words, hidden_size)
decoder = LuongAttnDecoderRNN(attn_model, output_embedding, hidden_size, output_sequence.n_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device

searcher = GreedySearchDecoder(encoder, decoder)


def eva(encoder, decoder, searcher, input_sequence, output_sequence):
        pair = random.choice(pairs)
        input_sentence = pair[0]

        input_sentence = inputNormalizeString(input_sentence)

        output_words = evaluate(encoder, decoder, searcher, input_sequence, output_sequence, input_sentence)
        # # Format and print response sentence
        output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD' or x == 'SOS')]
        
        reference = [output_words]
        candidate = normalizeString(pair[1])

        candidate = indexesFromSentence(output_sequence, candidate)

        candidate = [output_sequence.index2word[token] for token in candidate]

        candidate[:] = [x for x in candidate if not (x == 'EOS' or x == 'PAD' or x == 'SOS')]

        #BLEU4
        score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))

        print(score)

        return score

total_bleu_test_score = 0
number = 3000

for i in range(3000):
    score = eva(encoder, decoder, searcher, input_sequence, output_sequence)

    total_bleu_test_score += score

print("total_score: {}".format(total_bleu_test_score / number))

print('finish evaluate!')



######################################################################
# Run Evaluation
# ~~~~~~~~~~~~~~
#
# To chat with your model, run the following block.
#

# Set dropout layers to eval mode

