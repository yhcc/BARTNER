
from fastNLP import LossBase
import torch.nn.functional as F
from fastNLP import seq_len_to_mask


class Seq2SeqLoss(LossBase):
    def __init__(self):
        super().__init__()

    def get_loss(self, tgt_tokens, tgt_seq_len, pred):
        """

        :param tgt_tokens: bsz x max_len, 包含了的[sos, token, eos]
        :param pred: bsz x max_len-1 x vocab_size
        :return:
        """
        tgt_seq_len = tgt_seq_len - 1
        mask = seq_len_to_mask(tgt_seq_len, max_len=tgt_tokens.size(1) - 1).eq(0)
        tgt_tokens = tgt_tokens[:, 1:].masked_fill(mask, -100)
        loss = F.cross_entropy(target=tgt_tokens, input=pred.transpose(1, 2))
        return loss

