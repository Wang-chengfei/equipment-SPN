import torch.nn as nn
import torch
import torch.nn.functional as F
from models.set_decoder import SetDecoder
from models.set_criterion import SetCriterion
from models.seq_encoder import SeqEncoder
from utils.functions import generate_triple
import copy


class SetPred4RE(nn.Module):

    def __init__(self, args, num_classes):
        super(SetPred4RE, self).__init__()
        self.args = args
        self.encoder = SeqEncoder(args)
        config = self.encoder.config
        config.hidden_size = 768
        self.num_classes = num_classes
        self.decoder = SetDecoder(config, args.num_generated_triples, args.num_decoder_layers, num_classes,
                                  return_intermediate=False)
        self.criterion = SetCriterion(num_classes, loss_weight=self.get_loss_weight(args), na_coef=args.na_rel_coef,
                                      losses=["entity", "relation"], matcher=args.matcher)
        self.rel_weight = torch.ones(self.num_classes + 1).cuda()
        self.rel_weight[-1] = args.na_rel_coef

    def forward(self, input_ids, attention_mask, targets=None):
        last_hidden_state, pooler_output = self.encoder(input_ids, attention_mask)
        class_logits, head_start_logits, head_end_logits, tail_start_logits, tail_end_logits = self.decoder(
            encoder_hidden_states=last_hidden_state, pooler_output=pooler_output, encoder_attention_mask=attention_mask)
        # head_start_logits, head_end_logits, tail_start_logits, tail_end_logits = span_logits.split(1, dim=-1)
        head_start_logits = head_start_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(),
                                                                      -10000.0)
        head_end_logits = head_end_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        tail_start_logits = tail_start_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(),
                                                                      -10000.0)
        tail_end_logits = tail_end_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(),
                                                                  -10000.0)  # [bsz, num_generated_triples, seq_len]
        outputs = {'pred_rel_logits': class_logits, 'head_start_logits': head_start_logits,
                   'head_end_logits': head_end_logits, 'tail_start_logits': tail_start_logits,
                   'tail_end_logits': tail_end_logits}
        if targets is not None:
            loss = self.criterion(outputs, targets)
            # loss = self.my_criterion(outputs, targets)
            return loss, outputs
        else:
            return outputs

    def gen_triples(self, input_ids, attention_mask, info):
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            # print(outputs)
            pred_triple = generate_triple(outputs, info, self.args, self.num_classes)
            # print(pred_triple)
        return pred_triple

    def batchify(self, batch_list):
        batch_size = len(batch_list)
        sent_idx = [ele[0] for ele in batch_list]
        sent_ids = [ele[1] for ele in batch_list]
        targets = [ele[2] for ele in batch_list]
        sent_lens = list(map(len, sent_ids))
        max_sent_len = max(sent_lens)
        input_ids = torch.zeros((batch_size, max_sent_len), requires_grad=False).long()
        attention_mask = torch.zeros((batch_size, max_sent_len), requires_grad=False, dtype=torch.float32)
        for idx, (seq, seqlen) in enumerate(zip(sent_ids, sent_lens)):
            input_ids[idx, :seqlen] = torch.LongTensor(seq)
            attention_mask[idx, :seqlen] = torch.FloatTensor([1] * seqlen)
        if self.args.use_gpu:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            targets = [{k: torch.tensor(v, dtype=torch.long, requires_grad=False).cuda() for k, v in t.items()} for t in
                       targets]
        else:
            targets = [{k: torch.tensor(v, dtype=torch.long, requires_grad=False) for k, v in t.items()} for t in
                       targets]
        info = {"seq_len": sent_lens, "sent_idx": sent_idx}
        return input_ids, attention_mask, targets, info

    def my_criterion(self, outputs, targets):
        pred_rel = outputs['pred_rel_logits']
        pred_head_start = outputs['head_start_logits']
        pred_head_end = outputs['head_end_logits']
        pred_tail_start = outputs['tail_start_logits']
        pred_tail_end = outputs['tail_end_logits']

        target_rel = []
        target_head_start = []
        target_head_end = []
        target_tail_start = []
        target_tail_end = []
        # 扩充targets
        for target in targets:
            target['relation'] = F.pad(target['relation'],
                                       (0, self.args.num_generated_triples - target['relation'].numel()),
                                       mode='constant', value=self.num_classes)
            target['head_start_index'] = F.pad(target['head_start_index'],
                                               (0, self.args.num_generated_triples - target['head_start_index'].numel()),
                                               mode='constant', value=0)
            target['head_end_index'] = F.pad(target['head_end_index'],
                                             (0, self.args.num_generated_triples - target['head_end_index'].numel()),
                                             mode='constant', value=0)
            target['tail_start_index'] = F.pad(target['tail_start_index'],
                                               (0, self.args.num_generated_triples - target['tail_start_index'].numel()),
                                               mode='constant', value=0)
            target['tail_end_index'] = F.pad(target['tail_end_index'],
                                             (0, self.args.num_generated_triples - target['tail_end_index'].numel()),
                                             mode='constant', value=0)
            target_rel.append(target['relation'])
            target_head_start.append(target['head_start_index'])
            target_head_end.append(target['head_end_index'])
            target_tail_start.append(target['tail_start_index'])
            target_tail_end.append(target['tail_end_index'])

        # 合并targets
        target_rel = torch.stack(target_rel, dim=0)
        target_head_start = torch.stack(target_head_start, dim=0)
        target_head_end = torch.stack(target_head_end, dim=0)
        target_tail_start = torch.stack(target_tail_start, dim=0)
        target_tail_end = torch.stack(target_tail_end, dim=0)

        # 计算loss
        ent_weight = torch.ones(pred_head_start.size()[2]).cuda()
        ent_weight[0] = self.args.na_rel_coef
        rel_loss = F.cross_entropy(pred_rel.permute(0, 2, 1), target_rel, weight=self.rel_weight)
        head_start_loss = F.cross_entropy(pred_head_start.permute(0, 2, 1), target_head_start, weight=ent_weight)
        head_end_loss = F.cross_entropy(pred_head_end.permute(0, 2, 1), target_head_end, weight=ent_weight)
        tail_start_loss = F.cross_entropy(pred_tail_start.permute(0, 2, 1), target_tail_start, weight=ent_weight)
        tail_end_loss = F.cross_entropy(pred_tail_end.permute(0, 2, 1), target_tail_end, weight=ent_weight)

        loss_weight = self.get_loss_weight(self.args)
        loss = rel_loss * loss_weight["relation"] + (head_start_loss + head_end_loss) * loss_weight[
            "head_entity"] / 2 + (tail_start_loss + tail_end_loss) * loss_weight["tail_entity"] / 2
        return loss

    @staticmethod
    def get_loss_weight(args):
        return {"relation": args.rel_loss_weight, "head_entity": args.head_ent_loss_weight,
                "tail_entity": args.tail_ent_loss_weight}
