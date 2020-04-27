from torchtext import data, vocab
# from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

def load_data(file, g_sequence_len, embed_file):
    TEXT = data.Field(lower=True, fix_length=g_sequence_len, batch_first=True, eos_token='<eos>', init_token='<sos>')
    tb = data.TabularDataset(file, format='csv', fields=[('text', TEXT)])
    TEXT.build_vocab(tb, vectors=vocab.Vectors(embed_file), min_freq=5)
    return data_iterator, TEXT, tb

def load_data_2(file, g_sequence_len, embed_file):
    TEXT = data.Field(lower=True, fix_length=g_sequence_len, batch_first=True, eos_token='<eos>', init_token='<sos>')
    LABEL = data.Field(sequential=False)
    tb = data.TabularDataset(file, format='tsv', fields=[('text', TEXT), ('label', LABEL)])
    TEXT.build_vocab(tb, vectors=vocab.Vectors(embed_file), min_freq=5)
    # TEXT.build_vocab(tb, min_freq=5)
    LABEL.build_vocab(tb)
    label_names = LABEL.vocab.itos[1:]
    label_examples = [[] for _ in label_names]
    for each in tb:
        label_examples[label_names.index(each.label)].append(each)
    label_datasets = [data.Dataset(label_examples[i], fields=[('text', TEXT)]) for i in range(len(label_names))]

    return tb, TEXT, LABEL, label_names, label_datasets

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
        # combined += sentence_bleu(all, gs)
        combined += 0
    return combined/count
