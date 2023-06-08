import os
from typing import Dict, List, Optional

import torch
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from transformers import AutoModel, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

LLM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEVICE = LLM_DEVICE
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE
init_llm = "ChatGLM-6B"
init_llm = init_llm
init_embedding_model = "text2vec-base"
init_embedding_model = init_embedding_model

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class ChatLLM(LLM):
    max_token: int = 10000
    temperature: float = 0.1
    top_p = 0.9
    history = []
    model_type: str = "chatglm"
    model_name_or_path: str = init_llm,
    tokenizer: object = None
    model: object = None

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatLLM"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:

        response, _ = self.model.chat(
            self.tokenizer,
            prompt,
            history=self.history,
            max_length=self.max_token,
            temperature=self.temperature,
        )
        torch_gc()
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        self.history = self.history + [[None, response]]
        return response


    def load_llm(self,
                   llm_device=DEVICE,
                   **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path,trust_remote_code=True)
        if torch.cuda.is_available() and llm_device.lower().startswith("cuda"):
            self.model = (AutoModel.from_pretrained(
                self.model_name_or_path, trust_remote_code=True, **kwargs).quantize(4).half().cuda())
        else:
            self.model = (AutoModel.from_pretrained(self.model_name_or_path,trust_remote_code=True).float())
        self.model = self.model.eval()

            