import torch

from typing import Dict, Any, List, Optional
import torch
from typing import Optional, Dict, Any
from transformers import PreTrainedTokenizerFast

class Dataset(object):
    def skip(self, count: int):
        raise NotImplementedError()

    def fetch(self, batch: Optional[int] = None) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def where(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def assign(self, where: Dict[str, Any]):
        raise NotImplementedError()
from typing import Union

class TokenizedCorpus(Dataset):
    def __init__(self,
                 corpus_path: str,
                 seq_len: int,
                 tokenizer:PreTrainedTokenizerFast,
                 repeat: bool = True):
        self.corpus_fp = open(corpus_path, 'r', encoding='utf-8')
        self.seq_len = seq_len
        self.repeat = repeat
        self.tokenizer = tokenizer

    def skip(self, count: int):
        for _ in range(count):
            if not self.corpus_fp.readline():
                # Raise error when all sequences are fetched.
                if not self.repeat:
                    raise StopIteration()

                # Or, move to the first of the corpus.
                self.corpus_fp.seek(0)
                self.corpus_fp.readline()

    def _fetch_one(self) -> Dict[str, List[int]]:
        
        while True:
            # Read subword-tokenized sequence from corpus.
            line = self.corpus_fp.readline()
            if not line:
                # Raise error when all sequences are fetched.
                if not self.repeat:
                    raise StopIteration()

                # Or, move to the first of the corpus.
                self.corpus_fp.seek(0)
                continue

            # Use token indices rather than the token names directly.
            # print()
            indices = self.tokenizer.encode(line)
            if len(indices) + 2 > self.seq_len:
                continue

            # Decorate the sequence with additional tokens.
            print(type(self.tokenizer.bos_token_id))
            indices = [self.tokenizer.bos_token_id] + indices + [self.tokenizer.eos_token_id]
            indices += [self.tokenizer.pad_token_id] * (self.seq_len - len(indices) + 1)

            return {'input': indices[:-1], 'output': indices[1:]}

    def fetch(self, batch: Optional[int] = None) -> Dict[str, torch.Tensor]:
        if batch is None:
            data = self._fetch_one()
        else:
            data = [self._fetch_one() for _ in range(batch)]
            data = {k: [d[k] for d in data] for k in data[0]}

        return {k: torch.tensor(v, dtype=torch.long) for k, v in data.items()}

    def where(self) -> Dict[str, Any]:
        return {'offset': self.corpus_fp.tell()}

    def assign(self, where: Dict[str, Any]):
        self.corpus_fp.seek(where['offset'])