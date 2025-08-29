from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator, pad_and_concat


logger = logging.getLogger(__name__)

try:
    import onnxruntime_genai as og
except ModuleNotFoundError:
    pass

LogLikelihoodInputs = tuple[tuple[str, str], list[int], list[int]]

@register_model("onnx")
class LMEvalOnnxModelEvaluator(TemplateLM):
    def __init__(
        self,
        pretrained: str,
        batch_size: int | str = 1,
        max_length: int | None = None,
        ep: str = "follow_config",
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__()

        self.config = og.Config(pretrained)
        if ep != "follow_config":
            self.config.clear_providers()
            if ep != "cpu":
                self.config.append_provider(ep)
        self.model = og.Model(self.config)
        self.tokenizer = og.Tokenizer(self.model)

        # consider adding auto batch sizes
        self.batch_size = int(batch_size)
        
        if max_length:
            self.max_length = max_length
        else:
            with (Path(pretrained) / "genai_config.json").open() as f:
                self.max_length = json.load(f)["search"]["max_length"]
        self.params = og.GeneratorParams(self.model)
        self.params.set_search_options(max_length=self.max_length, past_present_share_buffer=False)
        
        self.device = device

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        """Tokenize a string using the model's tokenizer and return a list of token IDs."""
        return self.tokenizer.encode(string).tolist()

    def _model_call(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, _ = input_ids.shape
        self.params.set_search_options(batch_size=batch_size)
        generator = og.Generator(self.model, self.params)
        generator.append_tokens(input_ids.tolist())
        # [1, seq, vocab]
        return torch.from_numpy(generator.get_output("logits"))

    def _loglikelihood_tokens(self, requests: list[LogLikelihoodInputs], **kwargs) -> list[tuple[float, bool]]:
        def _collate(req: LogLikelihoodInputs):
            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        def _lookup_one_token_cont(req: LogLikelihoodInputs):
            return req[-2] + req[-1][:-1]

        re_ord = Collator(
            requests,
            sort_fn=_collate,
            group_by="contexts",
            group_fn=_lookup_one_token_cont,
        )

        res = []

        pbar = tqdm(
            total=len(re_ord),
            desc="Running loglikelihood requests"
        )
        logger.info(f"Calculating loglikelihood for {len(re_ord)} requests")
        for chunk in re_ord.get_batched(n=self.batch_size):
            inps = []
            cont_toks_list = []
            inplens = []

            padding_len_inp = None

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                total_length = len(context_enc) + len(continuation_enc)
                if total_length > self.max_length + 1:
                    logger.warning(
                        f"Combined length of context ({len(context_enc)}) and continuation ({len(continuation_enc)}) "
                        f"exceeds model's maximum length ({self.max_length}). "
                        f"Truncating {total_length - self.max_length + 1} tokens from the left."
                    )
                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                    dtype=torch.long,
                )
                (inplen,) = inp.shape

                padding_len_inp = (
                    max(padding_len_inp, inplen)
                    if padding_len_inp is not None
                    else inplen
                )

                inps.append(inp)  # [1, inp_length]
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

            batched_inps = pad_and_concat(
                padding_len_inp, inps, padding_side="right"
            )  # [batch, padding_len_inp]

            multi_logits = self._model_call(batched_inps) # [batch, padding_length (inp or cont), vocab]

            for (request_str, ctx_tokens, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):
                # Slice to original seq length
                contlen = len(cont_toks)
                # take only logits in the continuation
                # (discard context toks if decoder-only ; discard right-padding)
                # also discards + checks for "virtual tokens" in the causal LM's input window
                # from prompt/prefix tuning tokens, if applicable
                ctx_len = inplen + (logits.shape[0] - padding_len_inp)
                logits = logits[ctx_len - contlen : ctx_len]
                logits = logits.unsqueeze(0) # [1, seq, vocab]
                logits = F.log_softmax(logits.to(self.device), dim=-1)

                greedy_tokens = logits.argmax(dim=-1)

                for request_str, cont_toks, logits in re_ord.get_cache(  # noqa
                    req_str=request_str,
                    cxt_toks=ctx_tokens,
                    cont_toks=cont_toks,
                    logits=logits,
                ):
                    cont_toks = torch.tensor(
                        cont_toks, dtype=torch.long, device=self.device
                    ).unsqueeze(0)  # [1, seq]
                    # Use trailing slice [-cont_toks.shape[1]:] to handle variable length cont_len (but same ctx+cont[:-1]).
                    # i.e. continuations can be sliced at diff points. Collator ensures we have sufficient greedy_tokens
                    # by choosing key with longest cont if group_by="contexts".
                    max_equal = (
                        greedy_tokens[:, -cont_toks.shape[1] :] == cont_toks
                    ).all()

                    # Obtain log-probs at the corresponding continuation token indices
                    # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                        -1
                    )  # [1, seq]

                    # Answer: (log prob, is-exact-match)
                    answer = (float(logits.sum()), bool(max_equal))

                    res.append(answer)

                    if request_str is not None:
                        # special case: loglikelihood_rolling produces a number of loglikelihood requests
                        # all with cache key None. instead do add_partial on the per-example level
                        # in the loglikelihood_rolling() function for those.
                        self.cache_hook.add_partial(
                            "loglikelihood", request_str, answer
                        )
                    pbar.update(1)

        pbar.close()

        return re_ord.get_original(res)

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
        raise NotImplementedError("Yet to be implemented!")

    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        raise NotImplementedError("Yet to be implemented!")