from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional, List
import torch
from torch import Tensor
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.models.transformer import TransformerDecoder
from fairseq.models.transformer.transformer_base import (
    TransformerConfig,
    TransformerModelBase)
from fairseq.models import register_model

from entmax import Sparsemax, Entmax15, EntmaxBisect


@dataclass
class TransformerSparseOutConfig(TransformerConfig):
    proba_mapping_alpha: Optional[str] = field(
        default="1",
        metadata={"help": "entmax mapping alpha for test time"}
    )


class TransformerSparseOutDecoder(TransformerDecoder):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn, output_projection)

        proba_mapping_alpha = args.proba_mapping_alpha
        if proba_mapping_alpha == "1":
            self.proba_mapping = torch.nn.Softmax(dim=-1)
        elif proba_mapping_alpha == "1.5":
            self.proba_mapping = Entmax15(dim=-1)
        elif proba_mapping_alpha == "2":
            self.proba_mapping = Sparsemax(dim=-1)
        else:
            self.proba_mapping = EntmaxBisect(
                alpha=float(proba_mapping_alpha),
                n_iter=25,
                dim=-1)

    def get_normalized_probs_scriptable(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""

        if hasattr(self, "adaptive_softmax"):
            assert self.adaptive_softmax is None

        logits = net_output[0]
        probas = self.proba_mapping(logits)
        # fairseq decoder chokes with zeros,
        # so we add some tiny amount i guess.
        probas = torch.where(probas > 0, probas, 1e-8)
        if log_probs:
            log_probas = torch.log(probas)
            return log_probas
        else:
            return probas


@register_model("transformer_sparse_out", dataclass=TransformerSparseOutConfig)
class TransformerSparseOut(TransformerModelBase):

    @classmethod
    def add_args(cls, parser):
        gen_parser_from_dataclass(
            parser, TransformerSparseOutConfig(),
            delete_default=False, with_prefix=""
        )

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        decoder = TransformerSparseOutDecoder(
            cfg,
            tgt_dict,
            embed_tokens
        )
        return decoder
