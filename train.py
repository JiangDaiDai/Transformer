import torch
from torch import nn
from torch.utils import data

from model import load_data, load_array
from model import Encoder, Decoder, Transformer, trainModel

num_hiddens=32
num_layers = 2
dropout = 0.1 
batch_size = 1024 
num_steps = 32
lr = 0.001
num_epochs = 15

ffn_num_input = 32 
ffn_num_hiddens = 64
num_heads = 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

text_en = None
with open('./data/train.en', 'r', encoding='utf-8') as f:
    text_en = f.read()
text_en_bpe = text_en.replace('@@', '')
text_en_list = text_en_bpe.split('\n')

text_de = None
with open('./data/train.de', 'r', encoding='utf-8') as f:
    text_de = f.read()
text_de_bpe = text_de.replace('@@', '')
text_de_list = text_de_bpe.split('\n')
text_de_list = [item.split(' ') for item in text_de_list]
text_en_list = [item.split(' ') for item in text_en_list]
index2word = ['<unk>', '<pad>', '<bos>', '<eos>']
word2index = dict()
text_bpe = None
with open('./data/bpevocab', 'r', encoding='utf-8') as f:
    text_bpe = f.read()
text_bpe_list = [item.split(' ')[0] for item in text_bpe.split('\n')]
index2word = index2word + text_bpe_list
for i in range(len(index2word)):
    word2index[index2word[i]] = i

def vocab(Sequence, Dict = word2index):
    if isinstance(Sequence, str):
        if Sequence not in Dict : return Dict['<unk>']
        else : return Dict[Sequence]  
    res = []
    for item in Sequence:
        if item in Dict : res.append(Dict[item])
        else : res.append(Dict['<unk>'])
    return res  
src_array, src_valid_len = load_data(text_en_list, vocab, num_steps)
tgt_array, tgt_valid_len = load_data(text_de_list, vocab, num_steps)

data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
data_iter = load_array(data_arrays, batch_size)

encoder = Encoder(
    len(index2word), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
decoder = Decoder(
    len(index2word), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
net = Transformer(encoder, decoder)

trainModel(net, data_iter, lr, num_epochs, word2index, device)