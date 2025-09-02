from __future__ import annotations

import json
import logging
from pathlib import Path
import re
import gc

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator, pad_and_concat


logger = logging.getLogger(__name__)

LogLikelihoodInputs = tuple[tuple[str, str], list[int], list[int]]
TOKENIZER_INFINITY = 1000000000000000019884624838656

@register_model("ort")
class LMEvalORTEvaluator(TemplateLM):
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        model_path: str,
        batch_size: int | str = 1,
        max_length: int | None = None,
        ep: str | None = None,
        add_bos_token: bool | None = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.prefill = Prefill(model_path, ep)
        self.config = AutoConfig.from_pretrained(Path(model_path).parent)
        self.tokenizer = AutoTokenizer.from_pretrained(Path(model_path).parent)

        self.add_bos_token = add_bos_token

        # consider adding auto batch sizes
        self.batch_size = int(batch_size)
        self._max_length = max_length

    @property
    def max_length(self) -> int:
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.config, attr):
                return getattr(self.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == TOKENIZER_INFINITY:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def eot_token_id(self) -> int:
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    def tok_encode(
        self,
        string: str,
        left_truncate_len: int | None = None,
        add_special_tokens: bool | None = None,
    ) -> list[int]:
        special_tokens_kwargs = {}

        # by default for CausalLM - false or self.add_bos_token is set
        if add_special_tokens is None:
            special_tokens_kwargs = {
                "add_special_tokens": False or self.add_bos_token
            }
        # otherwise the method explicitly defines the value
        else:
            special_tokens_kwargs = {"add_special_tokens": add_special_tokens}

        encoding = self.tokenizer.encode(string, **special_tokens_kwargs)

        return encoding

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

        max_length = -1
        for _, context_enc, continuation_enc in requests:
            max_length = max(max_length, len(context_enc) + len(continuation_enc))
        max_length = min(max_length, self.max_length)
        print(max_length)
        self.prefill.initialize_buffers(self.batch_size, max_length)

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

            multi_logits = self.prefill.run(batched_inps) # [batch, padding_length (inp or cont), vocab]

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
                logits = F.log_softmax(logits, dim=-1)

                greedy_tokens = logits.argmax(dim=-1)

                for request_str, cont_toks, logits in re_ord.get_cache(  # noqa
                    req_str=request_str,
                    cxt_toks=ctx_tokens,
                    cont_toks=cont_toks,
                    logits=logits,
                ):
                    cont_toks = torch.tensor(
                        cont_toks, dtype=torch.long, device=greedy_tokens.device
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
        self.prefill.reset_buffers()

        return re_ord.get_original(res)

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
        raise NotImplementedError("Yet to be implemented!")

    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        raise NotImplementedError("Yet to be implemented!")

class Prefill:
    def __init__(self, model_path: str, ep: str):
        import onnx 
        from olive.model.utils.onnx_utils import get_additional_file_path, get_io_config

        self.model_path = model_path
        self.io_config = get_io_config(onnx.load(self.model_path, load_external_data=False))
        if ep == "CUDAExecutionProvider" and not torch.cuda.is_available():
            raise RuntimeError("CUDAExecutionProvider requires torch.cuda to be available")
        self.ep = ep
        self.device = "cuda" if ep == "CUDAExecutionProvider" else "cpu"

        self.io_dtypes = self.get_io_dtypes(self.io_config)
        self.vocab = dict(zip(self.io_config["output_names"], self.io_config["output_shapes"]))["logits"][-1]
        self.kv_info = self.get_kv_info(self.io_config)
        if self.kv_info is None:
            raise ValueError("Invalid io_config: kv_info not found")

        self._session = None
        self._batch_size = None
        self._buffers = None

    def run(self, input_ids: torch.Tensor) -> torch.Tensor:
        io_binding = self.session.io_binding()

        batch_size, seqlen = input_ids.shape
        if batch_size > self._batch_size:
            raise ValueError(f"Invalid batch size: {batch_size}, max batch size: {self._batch_size}")

        # bind inputs
        inputs_to_bind = {
            "input_ids": (input_ids.to(device=self.device, dtype=getattr(torch, self.io_dtypes["input_ids"])).contiguous(), self.io_dtypes["input_ids"], (batch_size, seqlen)),
        }
        for name, shape in [("attention_mask", (batch_size, seqlen)), ("past_seq_len", (batch_size,1)), ("total_seq_len", (1,))]:
            if name not in self._buffers["inputs"]:
                continue
            # no need to slice extra elements: attention_mask is all 1s, other inputs have fixed shapes
            inputs_to_bind[name] = (self._buffers["inputs"][name], self.io_dtypes[name], shape)
        if "position_ids" in self._buffers["inputs"]:
            # need to reallocate since the position_ids tensor may be sliced
            inputs_to_bind["position_ids"] = (self._buffers["inputs"]["position_ids"][:batch_size, :seqlen].contiguous(), self.io_dtypes["position_ids"], (batch_size, seqlen))
        for name in self._buffers["kv_inputs"]:
            inputs_to_bind[name] = (self._buffers["kv_inputs"][name], self.kv_info["dtype"], (batch_size, self.kv_info["num_kv_heads"], 0, self.kv_info["head_size"]))
        for name, (buffer, dtype, shape) in inputs_to_bind.items():
            io_binding.bind_input(
                name,
                device_type=self.device,
                device_id=0,
                element_type=dtype,
                shape=shape,
                buffer_ptr=buffer.data_ptr()
            )

        # bind outputs
        outputs_to_bind = {
            # provide full buffer, will slice batch_size * seqlen * self.vocab elements after run
            "logits": (self._buffers["outputs"]["logits"], self.io_dtypes["logits"], (batch_size, seqlen, self.vocab)),
        }
        for name in self._buffers["kv_outputs"]:
            outputs_to_bind[name] = (self._buffers["kv_outputs"][name], self.kv_info["dtype"], (batch_size, self.kv_info["num_kv_heads"], seqlen, self.kv_info["head_size"]))
        for name, (buffer, dtype, shape) in outputs_to_bind.items():
            io_binding.bind_output(
                name,
                device_type=self.device,
                device_id=0,
                element_type=dtype,
                shape=shape,
                buffer_ptr=buffer.data_ptr()
            )

        io_binding.synchronize_inputs()
        self.session.run_with_iobinding(io_binding)
        io_binding.synchronize_outputs()

        return self._buffers["outputs"]["logits"][:batch_size*seqlen*self.vocab].view(batch_size, seqlen, self.vocab)

    @property
    def session(self):
        from onnxruntime import InferenceSession

        if self._session is not None:
            return self._session

        self._session = InferenceSession(self.model_path, providers=[self.ep] if self.ep else None)
        return self._session

    def reset_buffers(self):
        self._buffers = None
        self._batch_size = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def initialize_buffers(self, batch_size: int, max_length: int):
        self.reset_buffers()

        # inputs other than kv cache
        inputs = {
            "attention_mask": torch.ones(batch_size*max_length, dtype=getattr(torch, self.io_dtypes["attention_mask"]), device=self.device)
        }
        if self.io_dtypes["position_ids"] is not None:
            inputs["position_ids"] = torch.arange(max_length, dtype=getattr(torch, self.io_dtypes["position_ids"]), device=self.device).unsqueeze(0).expand(batch_size, -1)
        if self.io_dtypes["past_seq_len"] is not None:
            inputs["past_seq_len"] = torch.tensor(max_length-1, dtype=getattr(torch, self.io_dtypes["past_seq_len"]), device=self.device).unsqueeze(0).expand(batch_size, -1)
            inputs["total_seq_len"] = torch.tensor(max_length, dtype=getattr(torch, self.io_dtypes["total_seq_len"]), device=self.device)

        # outputs other than kv cache
        outputs = {
            "logits": torch.zeros(batch_size * max_length * self.vocab, dtype=getattr(torch, self.io_dtypes["logits"]), device=self.device)
        }

        # kv cache inputs
        kv_inputs = {
            name: torch.zeros(
                0,
                dtype=getattr(torch, self.kv_info["dtype"]),
                device=self.device 
            ) for name in self.kv_info["past_names"]
        }

        # kv cache outputs
        kv_outputs = {
            name: torch.zeros(
                batch_size * self.kv_info["num_kv_heads"] * max_length * self.kv_info["head_size"],
                dtype=getattr(torch, self.kv_info["dtype"]),
                device=self.device
            ) for name in self.kv_info["present_to_past"]
        }

        self._buffers = {
            "inputs": inputs,
            "outputs": outputs,
            "kv_inputs": kv_inputs,
            "kv_outputs": kv_outputs
        }
        self._batch_size = batch_size

    @staticmethod
    def get_io_dtypes(io_config: dict) -> dict:
        """Return a dictionary mapping input names to their data types."""
        all_types = dict(zip(io_config["input_names"] + io_config["output_names"], io_config["input_types"] + io_config["output_types"]))
        return {
            name: all_types.get(name, None) for name in ["input_ids", "attention_mask", "position_ids", "past_seq_len", "total_seq_len", "logits"]
        }

    @staticmethod
    def get_kv_info(io_config: dict) -> dict | None:
        """Return the kv_info dictionary containing information about past keys and values.

        :param io_config: A dictionary containing the input and output names and shapes.
        :return: A dictionary with keys "past_names", "present_to_past", "num_kv_heads", and "head_size".
            If no kv_info is found, returns None. Only dynamic shapes are accepted currently.
        """
        # assuming batch_size, num_kv_heads, past_seq_len, head_size
        kv_options = {
            r"past_key_values.(\d+).key": {
                "past_key": "past_key_values.%d.key",
                "past_value": "past_key_values.%d.value",
                "present_key": "present.%d.key",
                "present_value": "present.%d.value",
            },
            r"past_key_(\d+)": {
                "past_key": "past_key_%d",
                "past_value": "past_value_%d",
                "present_key": "present_key_%d",
                "present_value": "present_value_%d",
            },
        }

        # Find the format of the past keys and values
        # only accept dynamic shapes for now
        kv_format = None
        for idx, i_name in enumerate(io_config["input_names"]):
            for pattern in kv_options:
                if re.match(pattern, i_name) and not isinstance(io_config["input_shapes"][idx][2], int):
                    kv_format = pattern
                    break
            if kv_format:
                break

        if kv_format is None:
            return None

        # find the number of layers
        num_layers = 0
        for i_name in io_config["input_names"]:
            num_layers += int(re.match(kv_format, i_name) is not None)

        past_names = []
        present_to_past = {}
        for k in ["key", "value"]:
            past_names.extend([kv_options[kv_format][f"past_{k}"] % i for i in range(num_layers)])
            present_to_past.update(
                {
                    kv_options[kv_format][f"present_{k}"] % i: kv_options[kv_format][f"past_{k}"] % i
                    for i in range(num_layers)
                }
            )

        past_shape = io_config["input_shapes"][io_config["input_names"].index(past_names[0])]

        return {
            "past_names": past_names,
            "present_to_past": present_to_past,
            "num_kv_heads": past_shape[1],
            "head_size": past_shape[3],
            "dtype": io_config["input_types"][io_config["input_names"].index(past_names[0])]
        }