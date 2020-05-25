from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torchtext import data, vocab

def get_tokenized_samples(samples, max_seq_length, tokenizer):
    """
    we assume a function label_map that maps each label to an index or vector encoding. Could also be a dictionary.
    :param samples: we assume struct {.text, .label)
    :param max_seq_length: the maximal sequence length
    :param tokenizer: BERT tokenizer
    :return: list of features
    """

    features = []
    for sample in samples:
        textlist = sample.text.split(' ')
        labellist = sample.label
        tokens = []
        labels = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word) #tokenize word according to BERT
            tokens.extend(token)
            label = labellist[i]
            # fit labels to tokenized size of word
            for m in range(len(token)):
                if m == 0:
                    labels.append(label)
                else:
                    labels.append("X")
        # if we exceed max sequence length, cut sample
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]

        ntokens = []
        segment_ids = []
        label_ids = []
        # start with [CLS] token
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(label_map(["[CLS]"]))
        for i, token in enumerate(tokens):
            # append tokens
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map(labels[i]))
        # end with [SEP] token
        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(label_map(["[SEP]"]))
        # convert tokens to IDs
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        # build mask of tokens to be accounted for
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            # pad with zeros to maximal length
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append([0] * (len(label_list) + 1))

        features.append((input_ids,
                              input_mask,
                              segment_ids,
                              label_id))
    return features

class MyBertBasedModel(BertPreTrainedModel):
    """
    MyBertBasedModel inherits from BertPreTrainedModel which is an abstract class to handle weights initialization and
        a simple interface for downloading and loading pre-trained models.
    """

    def __init__(self, config, num_labels):
        super(MyBertBasedModel, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config) # basic BERT model
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # now you can implement any architecture that receives bert sequence output
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = MyLoss()
            # it is important to activate the loss only on un-padded inputs
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1, self.num_labels)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            return loss
        else:
            return logits


class Sample:
    all_labels = {}
    def __init__(self, pair):
        self.text = pair[0]
        self.label = pair[1]
        if self.label not in self.all_labels.keys():
            self.all_labels[self.label] = 0
        self.all_labels[self.label] += 1


# file = '../data/test.tsv'

def load_data(file):
    data_file = open(file, 'r')
    data = [Sample(text.strip().split('\t')[1:]) for text in data_file.readlines()]
    data_file.close()
    return data


max_seq_length = 20
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
train_samples = load_data('../data/friends_train.tsv')
train_tokenized_samples = get_tokenized_samples(train_samples, max_seq_length, tokenizer)
model = MyBertBasedModel.from_pretrained('bert-base-uncased', num_labels = len(train_samples[0].all_labels))
model.train()
for _ in range(n_epochs):
    for sample in train_tokenized_samples:
        input_ids, input_mask, segment_ids, label_ids = sample
        loss = model(input_ids, segment_ids, input_mask, label_ids)
        loss.backward()
        optimizer.step()
