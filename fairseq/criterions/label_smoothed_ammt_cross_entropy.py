# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
import torch  # <<fix>>
import torch.nn as nn  # <<fix>>
from torch.autograd import Variable # <<fix>>


def one_hot(seq_batch, depth):
    out = Variable(torch.zeros(seq_batch.size() + torch.Size([depth]))).cuda()
    dim = len(seq_batch.size())
    index = seq_batch.view(seq_batch.size() + torch.Size([1]))
    return out.scatter_(dim, index, 1)


def compute_label_loss(multi_label_loss, scores, targets, source, ignore_index, tgt_vocab_size=None):
    targets = targets.contiguous()  # label_len, bsz
    bsz, label_len = targets.size()
    source = source.view(bsz, -1)
    mask = torch.ne(source, ignore_index).float()
    mask_scores = mask.unsqueeze(-1) * scores

    sum_scores = mask_scores.sum(1)
    labels = one_hot(targets, tgt_vocab_size).sum(1)
    labels = torch.gt(labels, 0).float()
    label_loss = multi_label_loss(sum_scores, labels)
    return label_loss


def label_smoothed_nll_loss(lprobs, target, lprobs_caption, source, logits_label, label, epsilon, loss1_coeff, loss2_coeff,
                            multi_label_loss, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

    # <<fix>>, caption and labels
    caption_nll_loss = -lprobs_caption.gather(dim=-1, index=source)
    caption_smooth_loss = -lprobs_caption.sum(dim=-1, keepdim=True)
    label_loss = compute_label_loss(multi_label_loss, logits_label, label, source, ignore_index,
                                    tgt_vocab_size=lprobs.size(-1))

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
        pad_src_mask = source.eq(ignore_index)
        caption_nll_loss.masked_fill_(pad_src_mask, 0.)
        caption_smooth_loss.masked_fill_(pad_src_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        caption_nll_loss = caption_nll_loss.squeeze(-1)
        caption_smooth_loss = caption_smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
        caption_nll_loss = caption_nll_loss.sum()
        caption_smooth_loss = caption_smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    caption_loss = (1. - epsilon) * caption_nll_loss + eps_i * caption_smooth_loss
    loss = loss + loss1_coeff * caption_loss + loss2_coeff * label_loss
    return loss, nll_loss


@register_criterion('label_smoothed_ammt_cross_entropy') # <<fix>>
class LabelSmoothedAMMTCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing, loss1_coeff, loss2_coeff):  # <<fix>>
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.loss1_coeff = loss1_coeff  # <<fix>>
        self.loss2_coeff = loss2_coeff  # <<fix>>
        # <<fix>>
        weight = torch.ones(len(task.target_dictionary)).cuda()
        weight[task.target_dictionary.pad()] = 0
        self.multi_label_loss = nn.MultiLabelSoftMarginLoss(weight=weight, reduction='sum')

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on
        # <<fix>>
        parser.add_argument('--loss1-coeff', default=0., type=float, metavar='D',
                            help='the loss coefficient for the additional target BoWs/EOT sequence')
        parser.add_argument('--loss2-coeff', default=0., type=float, metavar='D',
                            help='the loss coefficient for the additional target BoWs/EOT sequence')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        # <<fix>>
        source = sample['net_input']['src_tokens'].view(-1, 1)
        label = sample['net_input']['label']
        lprobs_caption, logits_label = net_output[-2], net_output[-1]
        lprobs_caption = lprobs_caption.view(-1, lprobs_caption.size(-1))
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, lprobs_caption, source, logits_label, label, self.eps, self.loss1_coeff, self.loss2_coeff, self.multi_label_loss,
            ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training. """
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
