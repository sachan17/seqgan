from torchtext import data, vocab
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

def load_data(file, g_sequence_len, bs, embed_file):
    TEXT = data.Field(lower=True, fix_length=g_sequence_len, batch_first=True, eos_token='<eos>', init_token='<sos>')
    tb = data.TabularDataset(file, format='csv', fields=[('text', TEXT)])
    TEXT.build_vocab(tb, vectors=vocab.Vectors(embed_file), min_freq=5)
    data_iterator = data.Iterator(tb, batch_size=bs)
    return data_iterator, TEXT, tb

def load_data_2(file, g_sequence_len, bs, embed_file):
    # file = 'data.csv'
    # g_sequence_len = 17
    # bs = 10
    TEXT = data.Field(lower=True, fix_length=g_sequence_len, batch_first=True, eos_token='<eos>', init_token='<sos>')
    LABEL = data.Field(sequential=False)
    tb = data.TabularDataset(file, format='tsv', fields=[('text', TEXT), ('label', LABEL)])
    # TEXT.build_vocab(tb, vectors=vocab.Vectors(embed_file), min_freq=5)
    TEXT.build_vocab(tb, min_freq=5)
    LABEL.build_vocab(tb)
    data_iterator = data.Iterator(tb, batch_size=bs)
    return data_iterator, TEXT, tb

def convert_to_text(TEXT, list_of_sent):
    list_of_sent = list_of_sent.detach().cpu()
    decoded_sent = []
    for sent in list_of_sent:
        decoded_sent.append([TEXT.vocab.itos[i] for i in sent if i not in [0, 1, 2, 3]])
    return decoded_sent

def bleu_4(TEXT, corpus, generator, g_sequence_len, count=10):
    generated = convert_to_text(TEXT, generator.sample(count, g_sequence_len))
    all = list(corpus.text)
    combined = 0
    for gs in generated:
    #for gs in tqdm(generated, mininterval=2, desc=' - BLEU_4', leave=False):
        combined += sentence_bleu(all, gs)
    return combined/count
