from dataclasses import dataclass, field
from functools import partial

import torch

from torch.nn import functional as F

from omegaconf import II

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq import metrics

from entmax import (Entmax15Loss,
                    SparsemaxLoss,
                    SparsemaxBisectLoss,
                    EntmaxBisectLoss)

from losses import NsectCudaLoss

@dataclass
class EntmaxLossCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")

    loss_alpha: float = field(
        default=1.5,
        metadata={"help": "alpha value for entmax loss"}
    )


class _BaseEntmaxCriterion(FairseqCriterion):
    def __init__(self, task, loss_alpha, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        assert loss_alpha > 1


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        logits = net_output[0]
        logits = logits.view(-1, logits.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = self.criterion(
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none"
        )(logits, target)
        return loss, loss  # weird

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion("entmax_nsect", dataclass=EntmaxLossCriterionConfig)
class EntmaxBisectCriterion(_BaseEntmaxCriterion):
    def __init__(self, task, loss_alpha, sentence_avg):
        super().__init__(task, loss_alpha, sentence_avg)
        self.criterion = partial(NsectCudaLoss, alpha=loss_alpha)


@register_criterion("entmax_bisect", dataclass=EntmaxLossCriterionConfig)
class EntmaxBisectCriterion(_BaseEntmaxCriterion):
    def __init__(self, task, loss_alpha, sentence_avg):
        super().__init__(task, loss_alpha, sentence_avg)
        self.criterion = partial(EntmaxBisectLoss, alpha=loss_alpha)


@register_criterion("sparsemax_bisect", dataclass=EntmaxLossCriterionConfig)
class SparsemaxBisectCriterion(_BaseEntmaxCriterion):
    def __init__(self, task, loss_alpha, sentence_avg):
        super().__init__(task, loss_alpha, sentence_avg)
        self.criterion = SparsemaxBisectLoss


@register_criterion("entmax15_exact", dataclass=EntmaxLossCriterionConfig)
class Entmax15ExactCriterion(_BaseEntmaxCriterion):
    def __init__(self, task, loss_alpha, sentence_avg):
        super().__init__(task, loss_alpha, sentence_avg)
        self.criterion = Entmax15Loss


@register_criterion("sparsemax_exact", dataclass=EntmaxLossCriterionConfig)
class SparsemaxExactCriterion(_BaseEntmaxCriterion):
    def __init__(self, task, loss_alpha, sentence_avg):
        super().__init__(task, loss_alpha, sentence_avg)
        self.criterion = SparsemaxLoss

