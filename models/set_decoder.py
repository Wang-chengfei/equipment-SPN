import torch.nn as nn
import torch


class SetDecoder(nn.Module):
    def __init__(self, config, num_generated_triples, num_layers, num_classes, return_intermediate=False):
        super().__init__()
        self.return_intermediate = return_intermediate
        self.num_generated_triples = num_generated_triples
        self.transformerDecoderLayer = DecoderLayer(d_model=config.hidden_size, nhead=8, num_layers=num_layers)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.query_embed = nn.Embedding(num_generated_triples, config.hidden_size)

        self.head_start_metric_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.head_end_metric_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.tail_start_metric_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.tail_end_metric_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.head_start_metric_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.head_end_metric_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.tail_start_metric_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.tail_end_metric_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.decoder2class = nn.Linear(config.hidden_size, num_classes + 1)

        self.head_start_metric_3 = nn.Linear(config.hidden_size, 1, bias=False)
        self.head_end_metric_3 = nn.Linear(config.hidden_size, 1, bias=False)
        self.tail_start_metric_3 = nn.Linear(config.hidden_size, 1, bias=False)
        self.tail_end_metric_3 = nn.Linear(config.hidden_size, 1, bias=False)

        torch.nn.init.orthogonal_(self.head_start_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.head_end_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_start_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_end_metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.head_start_metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.head_end_metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_start_metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.tail_end_metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.query_embed.weight, gain=1)

    def forward(self, encoder_hidden_states, pooler_output, encoder_attention_mask):
        bsz = encoder_hidden_states.size()[0]
        hidden_states = self.query_embed.weight.unsqueeze(0).repeat(bsz, 1, 1)
        hidden_states = self.dropout(self.LayerNorm(hidden_states))
        hidden_states = self.transformerDecoderLayer(hidden_states, encoder_hidden_states, encoder_attention_mask)

        # hidden_states(batch_size, num_generated_triples=10, config.hidden_size=768)
        # encoder_hidden_states(batch_size, length, config.hidden_size=768)
        # pooler_output(batch_size, config.hidden_size=768)
        # head_start_logits(batch_size, num_generated_triples=10, length)

        class_logits = self.decoder2class(hidden_states)

        hidden_states = self.head_start_metric_1(hidden_states)
        head_start_logits = self.head_start_metric_3(
            torch.relu(hidden_states.unsqueeze(2) + self.head_start_metric_2(encoder_hidden_states).unsqueeze(1))).squeeze()

        hidden_states = self.head_end_metric_1(hidden_states)
        head_end_logits = self.head_end_metric_3(
            torch.relu(hidden_states.unsqueeze(2) + self.head_end_metric_2(encoder_hidden_states).unsqueeze(1))).squeeze()

        hidden_states = self.tail_start_metric_1(hidden_states)
        tail_start_logits = self.tail_start_metric_3(
            torch.relu(hidden_states.unsqueeze(2) + self.tail_start_metric_2(encoder_hidden_states).unsqueeze(1))).squeeze()

        hidden_states = self.tail_end_metric_1(hidden_states)
        tail_end_logits = self.tail_end_metric_3(
            torch.relu(hidden_states.unsqueeze(2) + self.tail_end_metric_2(encoder_hidden_states).unsqueeze(1))).squeeze()

        return class_logits, head_start_logits, head_end_logits, tail_start_logits, tail_end_logits


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(DecoderLayer, self).__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=2048,
                                                        activation="relu")
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

    def forward(self, src, memory, memory_mask):
        # hidden_states(batch_size, num_generated_triples=10, config.hidden_size=768)
        # encoder_hidden_states(batch_size, length, config.hidden_size=768)
        # output(batch_size, num_generated_triples=10, config.hidden_size=768)
        src = src.transpose(0, 1)
        memory = memory.transpose(0, 1)
        output = self.transformer_decoder(tgt=src, memory=memory)
        output = output.transpose(0, 1)
        return output
