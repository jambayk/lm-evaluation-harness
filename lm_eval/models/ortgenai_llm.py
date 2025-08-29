from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator


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
        max_length: int | None = None,
        ep: str = "follow_config",
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

        if max_length:
            self.max_length = max_length
        else:
            with (Path(pretrained) / "genai_config.json").open() as f:
                self.max_length = json.load(f)["search"]["max_length"]
        self.params = og.GeneratorParams(self.model)
        self.params.set_search_options(max_length=self.max_length, past_present_share_buffer=False)

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        """Tokenize a string using the model's tokenizer and return a list of token IDs."""
        return self.tokenizer.encode(string).tolist()

    def _model_call(self, input_ids: list[int]) -> torch.Tensor:
        generator = og.Generator(self.model, self.params)
        generator.append_tokens(input_ids)
        # [1, seq, vocab]
        return torch.from_numpy(generator.get_output("logits"))

    def _loglikelihood_tokens(self, requests: list[LogLikelihoodInputs], disable_tqdm: bool = False, **kwargs) -> list[tuple[float, bool]]:
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
            disable=disable_tqdm,
            desc="Running loglikelihood requests",
        )

        for chunk in re_ord.get_batched(n=1):
            request_str, ctx_tokens, cont_tokens = chunk[0]

            # sanity checkes
            assert len(ctx_tokens) > 0
            assert len(cont_tokens) > 0
            assert len(cont_tokens) <= self.max_length

            total_len = len(ctx_tokens) + len(cont_tokens)
            if total_len > self.max_length + 1:
                logger.warning(
                    f"Combined length of context ({len(ctx_tokens)}) and continuation ({len(cont_tokens)}) "
                    f"exceeds model's maximum length ({self.max_length}). "
                    f"Truncating {total_len - (self.max_length + 1)} tokens from the left."
                )
            # [total - 1]
            inp = (ctx_tokens + cont_tokens)[-self.max_length:][:-1]

            # [1, total - 1, vocab]
            multi_logits = F.log_softmax(self._model_call(inp), dim=-1, dtype=torch.float32)

            contlen = len(cont_tokens)
            # [1, contlen, vocab]
            cont_slice = multi_logits[:, -contlen:]
            # [1, contlen]
            greedy_tokens = cont_slice.argmax(dim=-1)

            for req_str, cont_toks, shared_logits in re_ord.get_cache(
                req_str=request_str,
                cxt_toks=ctx_tokens,
                cont_toks=cont_tokens,
                logits=cont_slice
            ):
                # [1, contlen]
                cont_t = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0)
                # use trailing slice since cont_t maybe be variable
                is_exact = (greedy_tokens[:, -cont_t.shape[-1]:] == cont_t).all()

                # [1, contlen]
                tok_lp = torch.gather(shared_logits, 2, cont_t.unsqueeze(-1)).squeeze(-1)
                answer = (float(tok_lp.sum()), bool(is_exact))
                res.append(answer)

                if req_str is not None:
                    self.cache_hook.add_partial("loglikelihood", req_str, answer)

                pbar.update(1)

        pbar.close()
        return re_ord.get_original(res)

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
        raise NotImplementedError("Yet to be implemented!")

    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        raise NotImplementedError("Yet to be implemented!")