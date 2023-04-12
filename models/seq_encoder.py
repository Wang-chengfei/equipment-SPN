import torch
import torch.nn as nn
from transformers import BertModel


class EncoderConfig:
    def __int__(self, hidden_size, hidden_dropout_prob, vocab_size, layer_norm_eps):
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.vocab_size = vocab_size
        self.layer_norm_eps = layer_norm_eps


class SeqEncoder(nn.Module):
    def __init__(self, args):
        super(SeqEncoder, self).__init__()
        self.args = args
        # 使用BERT作为编码器
        if self.args.use_bert:
            if self.args.do_lower_case:
                self.bert = BertModel.from_pretrained(args.uncased_bert_directory)
            else:
                self.bert = BertModel.from_pretrained(args.cased_bert_directory)
            if args.fix_bert_embeddings:
                self.bert.embeddings.word_embeddings.weight.requires_grad = False
                self.bert.embeddings.position_embeddings.weight.requires_grad = False
                self.bert.embeddings.token_type_embeddings.weight.requires_grad = False
            self.config = self.bert.config
        # 使用LSTM作为编码器
        else:
            self.config = EncoderConfig()
            self.config.hidden_size = 768
            self.config.hidden_dropout_prob = 0.1
            self.config.vocab_size = 28996
            self.config.layer_norm_eps = 1e-5
            self.embedding = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
            self.lstm = nn.LSTM(input_size=self.config.hidden_size,
                                hidden_size=(int(self.config.hidden_size / 2)), batch_first=True, bidirectional=True,
                                dropout=self.config.hidden_dropout_prob, num_layers=3)

    def forward(self, input_ids, attention_mask):
        # 使用BERT作为编码器
        if self.args.use_bert:
            bert_output = self.bert(input_ids, attention_mask=attention_mask)
            last_hidden_state = bert_output.last_hidden_state
            pooler_output = bert_output.pooler_output
            return last_hidden_state, pooler_output
        # 使用LSTM作为编码器
        else:
            inputs = self.embedding(input_ids)
            masked_inputs = inputs * attention_mask.unsqueeze(-1)
            lengths = attention_mask.sum(dim=1).to(torch.device('cpu')).long()
            packed_inputs = nn.utils.rnn.pack_padded_sequence(masked_inputs, lengths, batch_first=True,
                                                              enforce_sorted=False)
            packed_outputs, _ = self.lstm(packed_inputs)
            outputs, h_n = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True,
                                                            total_length=inputs.size(1))
            return outputs, h_n
