from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer, GenerationConfig, PreTrainedTokenizer


class EmbeddingModel(nn.Module):
    def __init__(self, model_name_or_path: str, device: str = "cuda") -> None:
        super(EmbeddingModel, self).__init__()
        self.model_name_or_path = model_name_or_path
        self.device = device
        # self.max_len = max_len

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModel.from_pretrained(self.model_name_or_path)

        if self.device == "cuda":
            self.model.to("cuda")
            
        
        

    def forward(
        self,
        text: List[str],
        max_len: int,
    ) -> torch.Tensor:
        if isinstance(text, str):
            text = [text]

        tokenizer_output = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        if self.device == "cuda":
            for i in tokenizer_output.keys():
                tokenizer_output[i] = tokenizer_output[i].cuda()
        model_output = self.model(**tokenizer_output)
        embedding_value = model_output.last_hidden_state[:, 0]
        return embedding_value


def build_source_text(x: str, tokenizer) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {"role": "user", "content": x},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return text




class EmbeddingModel4Qwen2(nn.Module):
    # copy from https://huggingface.co/intfloat/e5-mistral-7b-instruct
    def __init__(
        self, model_name_or_path: str, device: str = "cuda", max_length: int = 4096
    ) -> None:
        super(EmbeddingModel4Qwen2, self).__init__()
        self.max_length = max_length
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            # device_map="cuda:0",
            pad_token_id=self.tokenizer.pad_token_id,
            trust_remote_code=True,
        )
        # self.model.generation_config = GenerationConfig.from_pretrained(
        #     model_name_or_path, pad_token_id=self.tokenizer.pad_token_id
        # )
        if self.device == "cuda":
            self.model.to("cuda")

        if True:  # model_args.use_lora:
            # logging.warning("Loading model to Lora")

            from peft import LoraConfig, get_peft_model

            LORA_R = 32
            # LORA_ALPHA = 16
            LORA_DROPOUT = 0.05
            TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

            config = LoraConfig(
                r=LORA_R,
                # lora_alpha=LORA_ALPHA,
                target_modules=TARGET_MODULES,
                lora_dropout=LORA_DROPOUT,
                bias="none",
                task_type="FEATURE_EXTRACTION",  # "CAUSAL_LM",  # "FEATURE_EXTRACTION",
            )
            # model = model.to(torch.bfloat16)
            self.model = get_peft_model(self.model, config)
            # peft_module_casting_to_bf16(model)
            self.model.print_trainable_parameters()

    def forward(
        self,
        text: List[str],
        max_len: int = None,
        is_query: bool = True,
    ):
        all_raw_text = text

        if is_query:
            batch_raw_text = []
            for q in all_raw_text:
                raw_text = build_source_text(q, self.tokenizer)
                batch_raw_text.append(raw_text)
        else:
            if not isinstance(text, list):
                batch_raw_text = [all_raw_text]
            else:
                batch_raw_text = all_raw_text
            

        max_length = max_len if max_len is not None else self.max_length
        # Tokenize the input texts
        batch_dict = self.tokenizer(
            batch_raw_text,
            max_length=max_length - 1,
            return_attention_mask=False,
            padding=False,
            truncation=True,
        )
        # append eos_token_id to every input_ids
        batch_dict["input_ids"] = [
            input_ids + [self.tokenizer.eos_token_id]
            for input_ids in batch_dict["input_ids"]
        ]
        batch_dict = self.tokenizer.pad(
            batch_dict, padding=True, return_attention_mask=True, return_tensors="pt"
        )
        # batch_dict.pop("token_type_ids")

        for tempkey in batch_dict.keys():
            batch_dict[tempkey] = batch_dict[tempkey].to(self.model.device)

        modeloutput = self.model(**batch_dict)

        embeddings = self.last_token_pool(
            modeloutput.last_hidden_state, batch_dict["attention_mask"]
        )
        return embeddings

    def last_token_pool(
        self, last_hidden_states: Tensor, attention_mask: Tensor
    ) -> Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]


class EmbeddingModel4Qwen(nn.Module):
    #copy from https://huggingface.co/intfloat/e5-mistral-7b-instruct
    def __init__(
        self, model_name_or_path: str, device: str = "cuda", max_length: int = 4096
    ) -> None:
        super(EmbeddingModel4Qwen, self).__init__()
        self.max_length = max_length
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            pad_token="<|extra_0|>",
            eos_token="<|endoftext|>",
        )

        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            # device_map="cuda:0",
            pad_token_id=self.tokenizer.pad_token_id,
            trust_remote_code=True,
        )
        self.model.generation_config = GenerationConfig.from_pretrained(
            model_name_or_path, pad_token_id=self.tokenizer.pad_token_id
        )
        if self.device == "cuda":
            self.model.to("cuda")

    def forward(
        self,
        text: List[str],
        max_len: int = None,
        is_query:bool=True,
    ):
        all_raw_text = text
        
        if is_query:
            batch_raw_text = []
            for q in all_raw_text:
                raw_text, _ = make_context(
                    self.tokenizer,
                    q,
                    system="You are a helpful assistant.",
                    max_window_size=self.model.generation_config.max_window_size,
                    chat_format=self.model.generation_config.chat_format,
                )
                batch_raw_text.append(raw_text)
        else:
            batch_raw_text = all_raw_text

        max_length = max_len if max_len is not None else self.max_length
        # Tokenize the input texts
        batch_dict = self.tokenizer(
            batch_raw_text,
            max_length=max_length - 1,
            return_attention_mask=False,
            padding=False,
            truncation=True,
        )
        # append eos_token_id to every input_ids
        batch_dict["input_ids"] = [
            input_ids + [self.tokenizer.eos_token_id]
            for input_ids in batch_dict["input_ids"]
        ]
        batch_dict = self.tokenizer.pad(
            batch_dict, padding=True, return_attention_mask=True, return_tensors="pt"
        )
        batch_dict.pop("token_type_ids")

        for tempkey in batch_dict.keys():
            batch_dict[tempkey] = batch_dict[tempkey].to(self.model.device)

        modeloutput = self.model(**batch_dict)

        embeddings = self.last_token_pool(
            modeloutput.last_hidden_state, batch_dict["attention_mask"]
        )
        return embeddings

    def last_token_pool(
        self, last_hidden_states: Tensor, attention_mask: Tensor
    ) -> Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]


def make_context(
    tokenizer: PreTrainedTokenizer,
    query: str,
    history: List[Tuple[str, str]] = None,
    system: str = "",
    max_window_size: int = 6144,
    chat_format: str = "chatml",
):
    if history is None:
        history = []

    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}", tokenizer.encode(
                role, allowed_special=set()
            ) + nl_tokens + tokenizer.encode(content, allowed_special=set())

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            response_text, response_tokens_part = _tokenize_str(
                "assistant", turn_response
            )
            response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

            next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
            prev_chat = (
                f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
            )

            current_context_size = (
                len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            )
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (
            nl_tokens
            + im_start_tokens
            + _tokenize_str("user", query)[1]
            + im_end_tokens
            + nl_tokens
            + im_start_tokens
            + tokenizer.encode("assistant")
            + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

    return raw_text, context_tokens
