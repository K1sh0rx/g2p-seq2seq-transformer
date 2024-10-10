!nvidia-smi

!ls
import os

if not os.path.isdir('cmudict-master'):

    !pip -q install python-Levenshtein

    !wget -q "https://github.com/cmusphinx/cmudict/archive/master.zip"

    !unzip -q master.zip
import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

import time

import random

import math



random.seed(1488)

torch.manual_seed(1488)

torch.cuda.manual_seed(1488)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 1024



graphemes = ['PAD', 'SOS'] + list('abcdefghijklmnopqrstuvwxyz.\'-') + ['EOS']

with open('cmudict-master/cmudict.symbols', 'r') as f:

    phonemes = ['PAD', 'SOS'] + f.read().strip().split('\n') + ['EOS']



g2idx = {g: idx for idx, g in enumerate(graphemes)}

idx2g = {idx: g for idx, g in enumerate(graphemes)}



p2idx = {p: idx for idx, p in enumerate(phonemes)}

idx2p = {idx: p for idx, p in enumerate(phonemes)}



def g2seq(s):

    return [g2idx['SOS']] + [g2idx[i] for i in s if i in g2idx.keys()] + [g2idx['EOS']]



def seq2g(s):

    return [idx2g[i] for i in s if idx2g[i]]



def p2seq(s):

    return [p2idx['SOS']] + [p2idx[i] for i in s.split() if i in p2idx.keys()] + [p2idx['EOS']]



def seq2p(s):

    return [idx2p[i] for i in s]
import Levenshtein

def phoneme_error_rate(p_seq1, p_seq2):

    p_vocab = set(p_seq1 + p_seq2)

    p2c = dict(zip(p_vocab, range(len(p_vocab))))

    c_seq1 = [chr(p2c[p]) for p in p_seq1]

    c_seq2 = [chr(p2c[p]) for p in p_seq2]

    return Levenshtein.distance(''.join(c_seq1),

                                ''.join(c_seq2)) / len(c_seq2)
class TextLoader(torch.utils.data.Dataset):

    def __init__(self, path='cmudict-master/cmudict.dict'):

        self.x, self.y = [], []

        with open(path, 'r') as f:

            data = f.read().strip().split('\n')

        for line in data:

            x, y = line.split(maxsplit=1)

            self.x.append(g2seq(x))

            self.y.append(p2seq(y))



    def __getitem__(self, index):

        return (torch.LongTensor(self.x[index]), torch.LongTensor(self.y[index]))



    def __len__(self):

        return len(self.x)





class TextCollate():

    def __call__(self, batch):

        max_x_len = max([i[0].size(0) for i in batch])

        x_padded = torch.LongTensor(max_x_len, len(batch))

        x_padded.zero_()



        max_y_len = max([i[1].size(0) for i in batch])

        y_padded = torch.LongTensor(max_y_len, len(batch))

        y_padded.zero_()



        for i in range(len(batch)):

            x = batch[i][0]

            x_padded[:x.size(0), i] = x

            y = batch[i][1]

            y_padded[:y.size(0), i] = y



        return x_padded, y_padded
pin_memory = True

num_workers = 2



dataset = TextLoader()

train_len = int(len(dataset) * 0.9)

trainset, valset = torch.utils.data.random_split(dataset, [train_len, len(dataset) - train_len])



collate_fn = TextCollate()



train_loader = torch.utils.data.DataLoader(trainset, num_workers=num_workers, shuffle=True,

                          batch_size=batch_size, pin_memory=pin_memory,

                          drop_last=True, collate_fn=collate_fn)



val_loader = torch.utils.data.DataLoader(valset, num_workers=num_workers, shuffle=False,

                        batch_size=batch_size, pin_memory=pin_memory,

                        drop_last=False, collate_fn=collate_fn)
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):

        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        self.scale = nn.Parameter(torch.ones(1))



        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(

            0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)



    def forward(self, x):

        x = x + self.scale * self.pe[:x.size(0), :]

        return self.dropout(x)



class TransformerModel(nn.Module):

    def __init__(self, intoken, outtoken, hidden, enc_layers=3, dec_layers=1, dropout=0.1):

        super(TransformerModel, self).__init__()

        nhead = hidden//64



        self.encoder = nn.Embedding(intoken, hidden)

        self.pos_encoder = PositionalEncoding(hidden, dropout)



        self.decoder = nn.Embedding(outtoken, hidden)

        self.pos_decoder = PositionalEncoding(hidden, dropout)



        self.transformer = nn.Transformer(d_model=hidden, nhead=nhead, num_encoder_layers=enc_layers, num_decoder_layers=dec_layers, dim_feedforward=hidden*4, dropout=dropout, activation='relu')

        self.fc_out = nn.Linear(hidden, outtoken)



        self.src_mask = None

        self.trg_mask = None

        self.memory_mask = None



    def generate_square_subsequent_mask(self, sz):

        mask = torch.triu(torch.ones(sz, sz), 1)

        mask = mask.masked_fill(mask==1, float('-inf'))

        return mask



    def make_len_mask(self, inp):

        return (inp == 0).transpose(0, 1)



    def forward(self, src, trg):

        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):

            self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device)



        src_pad_mask = self.make_len_mask(src)

        trg_pad_mask = self.make_len_mask(trg)



        src = self.encoder(src)

        src = self.pos_encoder(src)



        trg = self.decoder(trg)

        trg = self.pos_decoder(trg)



        output = self.transformer(src, trg, src_mask=self.src_mask, tgt_mask=self.trg_mask, memory_mask=self.memory_mask,

                                  src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=trg_pad_mask, memory_key_padding_mask=src_pad_mask)

        output = self.fc_out(output)



        return output
INPUT_DIM = len(graphemes)

OUTPUT_DIM = len(phonemes)



model = TransformerModel(INPUT_DIM, OUTPUT_DIM, hidden=128, enc_layers=3, dec_layers=1).to(device)



def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)



print(f'The model has {count_parameters(model):,} trainable parameters')

print(model)
optimizer = optim.AdamW(model.parameters())



TRG_PAD_IDX = p2idx['PAD']



criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
def train(model, optimizer, criterion, iterator):



    model.train()



    epoch_loss = 0



    for i, batch in enumerate(iterator):

        src, trg = batch

        src, trg = src.cuda(), trg.cuda()



        optimizer.zero_grad()

        output = model(src, trg[:-1,:])

        loss = criterion(output.transpose(0, 1).transpose(1, 2), trg[1:,:].transpose(0, 1))

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        epoch_loss += loss.item()



    return epoch_loss / len(iterator)
def evaluate(model, criterion, iterator):



    model.eval()



    epoch_loss = 0



    with torch.no_grad():

        for i, batch in enumerate(iterator):

            src, trg = batch

            src, trg = src.cuda(), trg.cuda()



            output = model(src, trg[:-1,:])

            loss = criterion(output.transpose(0, 1).transpose(1, 2), trg[1:,:].transpose(0, 1))

            epoch_loss += loss.item()



        # for i in [0, 5, -1]:

        #     print('word:', ''.join(seq2g(src[:,i].tolist())).replace('PAD', '').replace('SOS', '').replace('EOS', ''))

        #     print('real p:    ', seq2p(trg[1:,i].tolist()))

        #     print('predict p: ', seq2p(output[:,i].argmax(-1).tolist()))



    return epoch_loss / len(iterator)
%%time

def epoch_time(start_time, end_time):

    elapsed_time = end_time - start_time

    elapsed_mins = int(elapsed_time / 60)

    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs



N_EPOCHS = 100



best_valid_loss = float('inf')



for epoch in range(N_EPOCHS):

    print(f'Epoch: {epoch+1:02}')



    start_time = time.time()



    train_loss = train(model, optimizer, criterion, train_loader)

    valid_loss = evaluate(model, criterion, val_loader)



    epoch_mins, epoch_secs = epoch_time(start_time, time.time())



    if valid_loss < best_valid_loss:

        best_valid_loss = valid_loss

        # torch.save(model.state_dict(), 'g2p_model.pt')



    print(f'Time: {epoch_mins}m {epoch_secs}s')

    print(f'Train Loss: {train_loss:.3f}')

    print(f'Val   Loss: {valid_loss:.3f}')

print(best_valid_loss)
from tqdm.notebook import tqdm

max_len = 50



val_loader = torch.utils.data.DataLoader(valset, num_workers=2, shuffle=False,

                                         batch_size=1, pin_memory=False,

                                         drop_last=False, collate_fn=collate_fn)



def validate(model, dataloader, show=10):

    model.eval()

    show_count = 0

    error_w = 0

    error_p = 0

    with torch.no_grad():

        for batch in tqdm(dataloader):

            src, trg = batch

            src, trg = src.cuda(), trg.cuda()

            real_p = seq2p(trg.squeeze(1).tolist())

            real_g = seq2g(src.squeeze(1).tolist()[1:-1])



            memory = model.transformer.encoder(model.pos_encoder(model.encoder(src)))



            out_indexes = [p2idx['SOS'], ]



            for i in range(max_len):

                trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(device)



                output = model.fc_out(model.transformer.decoder(model.pos_decoder(model.decoder(trg_tensor)), memory))

                out_token = output.argmax(2)[-1].item()

                out_indexes.append(out_token)

                if out_token == p2idx['EOS']:

                    break



            out_p = seq2p(out_indexes)

            error_w += int(real_p != out_p)

            error_p += phoneme_error_rate(real_p, out_p)

            if show > show_count:

                show_count += 1

                print('Real g', ''.join(real_g))

                print('Real p', real_p)

                print('Pred p', out_p)

    return error_p/len(dataloader)*100, error_w/len(dataloader)*100



print(validate(model, val_loader, show=10))

# model.load_state_dict(torch.load('g2p_model.pt'))

# print(validate(model, val_loader, show=10))
# print(model.pos_encoder.scale)

# print(model.pos_decoder.scale)
src = torch.tensor(g2seq('otorhinolaryngologist')).unsqueeze(1).cuda()



memory = model.transformer.encoder(model.pos_encoder(model.encoder(src)))



out_indexes = [p2idx['SOS'], ]



for i in range(max_len):

    trg_tensor = torch.LongTensor(out_indexes).unsqueeze(1).to(device)



    output = model.fc_out(model.transformer.decoder(model.pos_decoder(model.decoder(trg_tensor)), memory))

    out_token = output.argmax(2)[-1].item()

    out_indexes.append(out_token)

    if out_token == p2idx['EOS']:

        break

seq2p(out_indexes)
