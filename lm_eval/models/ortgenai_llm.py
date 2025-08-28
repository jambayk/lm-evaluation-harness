from __future__ import annotations

import logging

from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator
from tqdm import tqdm
from typing import Sequence

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
        batch_size: int | str | None = 1,
        max_batch_size: int | None = 64,
        ep: str = "follow_config"
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

        self.max_length = max_length if max_length is not None else self.config.search.max_length
        self.max_batch_size = max_batch_size or 64
        batch_size = batch_size or 1
        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size = int(batch_size)

        self.params = og.GeneratorParams(self.model)
        self.params.set_search_options(batch_size=self.batch_size, max_length=self.max_length)

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        """Tokenize a string using the model's tokenizer and return a list of token IDs."""
        return self.tokenizer.encode(string).tolist()

    def _model_call(self, input_ids: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        generator = og.Generator(self._model, self._params)
        generator.append_tokens(input_ids)

        count = input_ids.shape[0]
        with torch.no_grad():
            while not generator.is_done() and count < self._max_length:
                generator.generate_next_token()
                count += 1

            logits = generator.get_logits().squeeze().squeeze().tolist()
            tokens = generator.get_sequence(0)

        log_probs = torch.nn.functional.log_softmax(torch.tensor(logits), dim=-1).numpy()
        return logits, log_probs, tokens

    def _loglikelihood_tokens(self, requests: list[LogLikelihoodInputs], **kwargs) -> list[tuple[float, bool]]:
        def _collate(req: LogLikelihoodInputs):
            """Define the key for the sorted method."""
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        disable_tqdm = kwargs.get("disable_tqdm") or False

        result = []
        re_ord = Collator(requests, sort_fn=_collate, group_by=None)
        pbar = tqdm(desc="Running loglikelihood requests", total=len(requests), disable=disable_tqdm)
        for chunk in re_ord.get_batched(n=self._batch_size):
            _, context_enc, continuation_enc = next(iter(chunk))

            input_ids = (context_enc + continuation_enc)[-(self._max_length + 1) :][:-1]
            ctx_len = len(input_ids)

            if len(context_enc) + len(continuation_enc) > (self._max_length + 1):
                logger.warning(
                    "Context length (%d) + continuation length (%d) > max_length (%d). Left truncating context.",
                    len(context_enc),
                    len(continuation_enc),
                    self._max_length,
                )

            input_ids = np.asarray(input_ids)
            _, log_probs, output_tokens = self._model_call(input_ids)

            cont_len = len(continuation_enc)
            cont_tokens = np.asarray(continuation_enc)
            greedy_tokens = output_tokens[ctx_len - cont_len : ctx_len]

            is_greedy = (cont_tokens == greedy_tokens).all()
            log_probs = np.take(log_probs, cont_tokens, 0)

            answer = (float(log_probs.sum()), bool(is_greedy))
            result.append(answer)

            pbar.update(1)

        pbar.close()
        return re_ord.get_original(result)

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
        raise NotImplementedError("Yet to be implemented!")

    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        raise NotImplementedError("Yet to be implemented!")

    def _detect_batch_size(self, requests: Sequence | None = None, pos: int = 0):
        if requests:
            _, context_enc, continuation_enc = requests[pos]
            max_length = len(
                (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1]
            )
            max_context_enc = len(context_enc[-(self.max_length + 1) :])
            max_cont_enc = len(continuation_enc[-(self.max_length + 1) :])
        else:
            max_length = self.max_length
            max_context_enc = max_length
            max_cont_enc = max_length

        batch_size = self.max_batch_size

        def reduce_batch_size_fn():
            nonlocal batch_size
            batch_size = int(batch_size * 0.9)
            return batch_size

        while True:
            if batch_size == 0:
                return 1
            try:
                inputs = [[1]*length]


        

        # if OOM, then halves batch_size and tries again
        @find_executable_batch_size(starting_batch_size=self.max_batch_size)
        def forward_batch(batch_size: int):
            if self.backend == "seq2seq":
                length = max(max_context_enc, max_cont_enc)
                batched_conts = torch.ones(
                    (batch_size, length), device=self.device
                ).long()
                test_batch = torch.ones((batch_size, length), device=self.device).long()
                call_kwargs = {
                    "attn_mask": test_batch,
                    "labels": batched_conts,
                }
            else:
                call_kwargs = {}
                test_batch = torch.ones(
                    (batch_size, max_length), device=self.device
                ).long()
            for _ in range(5):
                out = F.log_softmax(  # noqa: F841
                    self._model_call(test_batch, **call_kwargs),
                    dim=-1,
                    dtype=self.softmax_dtype,
                )

            return batch_size

        try:
            batch_size = forward_batch()
        except RuntimeError as e:
            if "No executable batch size found" in str(e):
                batch_size = 1
            else:
                raise

        if self.world_size > 1:
            # if multi-GPU, always take minimum over all selected batch sizes
            max_rnk_bs = torch.tensor([batch_size], device=self.device)
            gathered = (
                self.accelerator.gather(max_rnk_bs).cpu().detach().numpy().tolist()
            )
            batch_size = min(gathered)
            clear_torch_cache()
            return batch_size

        clear_torch_cache()
        return batch_size