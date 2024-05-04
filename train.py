from data import TokenizedCorpus
from typing import Tuple, Iterator, Dict
import torch 
from torch import optim,nn
from transformers import PreTrainedTokenizerFast
from torch.utils.data import Dataset
from transformer import Transformer
class TrainingSpec(object):
    def initialize(self):
        pass

    def prepare_datasets(self) -> Tuple[Dataset, Dataset]:
        raise NotImplementedError()

    def construct_model(self) -> nn.Module:
        raise NotImplementedError()

    def create_optimizer(self, params: Iterator[nn.Parameter]
                         ) -> Tuple[optim.Optimizer,
                                    optim.lr_scheduler._LRScheduler]:
        raise NotImplementedError()

    def train_objective(self, data: Dict[str, torch.Tensor], model: nn.Module
                        ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def eval_objective(self, data: Dict[str, torch.Tensor], model: nn.Module
                       ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()
    
def create_tokenizer(tok_path):
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_path)
    special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]"]
    fast_tokenizer.eos_token = special_tokens[2]
    fast_tokenizer.pad_token = special_tokens[3]
    fast_tokenizer.bos_token = special_tokens[1]
    fast_tokenizer.unk_token = special_tokens[0]
    return fast_tokenizer

class GPT2TrainingSpec(TrainingSpec):
    def __init__(self, train_corpus: str, eval_corpus: str, tokenizer_path: str,
                 seq_len: int, layers: int, heads: int, dims: int, rate: int,
                 dropout: float, base_lr: float, wd_rate: float,
                 total_steps: int, use_grad_ckpt: bool):
        self.train_corpus = train_corpus
        self.eval_corpus = eval_corpus
        self.seq_len = seq_len
        self.layers = layers
        self.heads = heads
        self.dims = dims
        self.rate = rate
        self.dropout = dropout
        self.base_lr = base_lr
        self.wd_rate = wd_rate
        self.total_steps = total_steps
        self.use_grad_ckpt = use_grad_ckpt
        self.tokenizer_path =  tokenizer_path

    def initialize(self):
        self.tokenizer = create_tokenizer(self.tokenizer_path)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id,
                                             reduction='mean')

    def prepare_datasets(self) -> Tuple[Dataset, Dataset]:
        train_dataset = TokenizedCorpus(corpus_path=self.train_corpus,
                                        tokenizer=self.tokenizer,
                                        seq_len=self.seq_len)
        eval_dataset = TokenizedCorpus(corpus_path=self.eval_corpus,
                                       tokenizer=self.tokenizer,
                                       seq_len=self.seq_len)
        return train_dataset, eval_dataset

    def construct_model(self) -> nn.Module:
        return Transformer(layers=self.layers, pad_idx=self.tokenizer.pad_token_id,
                           words=len(self.tokenizer.vocab), seq_len=self.seq_len,
                           heads=self.heads, dims=self.dims, rate=self.rate,
                           dropout=self.dropout, bidirectional=False)

    def create_optimizer(self, params: Iterator[nn.Parameter]
                         ) -> Tuple[optim.Optimizer,
                                    optim.lr_scheduler._LRScheduler]:
        try:
            from apex.optimizers import FusedAdam as Adam
            from apex.normalization import FusedLayerNorm as LayerNorm
        except ModuleNotFoundError:
            from torch.optim import AdamW as Adam
            from torch.nn import LayerNorm
        optimizer = Adam(
            params, lr=self.base_lr, weight_decay=self.wd_rate)
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 1 - step / self.total_steps)
        return optimizer, scheduler

    def train_objective(self, data: Dict[str, torch.Tensor], model: nn.Module
                        ) -> Dict[str, torch.Tensor]:
        logits = model(data['input'], use_grad_ckpt=self.use_grad_ckpt)
        loss = self.criterion(logits.transpose(1, 2), data['output'])
        return {'loss': loss}

    def eval_objective(self, data: Dict[str, torch.Tensor], model: nn.Module
                       ) -> Dict[str, torch.Tensor]:
        logits, _ = model(data['input'], past=None)
        loss = self.criterion(logits.transpose(1, 2), data['output'])
        return {'loss': loss}


class TrainConfig(object):
    def __init__(self,
                 batch_train: int,
                 batch_eval: int,
                 total_steps: int,
                 eval_steps: int,
                 save_steps: int,
                 save_model_path: str,
                 save_checkpoint_path: str,
                 description: str,
                 log_format: str,
                 use_amp: bool,
                 gpus: int):
        self.batch_train = batch_train
        self.batch_eval = batch_eval
        self.total_steps = total_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.save_model_path = save_model_path
        self.save_checkpoint_path = save_checkpoint_path
        self.description = description
        self.log_format = log_format
        self.use_amp = use_amp
        self.gpus = gpus

    @property
    def distributed(self) -> bool:
        return self.gpus is not None and self.gpus > 1

from typing import Dict, Optional


class Recorder(object):
    def __init__(self):
        self.metrics = {}
        self.batch_metrics = {}

    def record(self, metrics: Dict[str, float], scope: Optional[str] = None):
        for name, value in metrics.items():
            name = f'{scope}/{name}' if scope else name

            if name not in self.batch_metrics:
                self.batch_metrics[name] = []
            self.batch_metrics[name].append(value)

    def stamp(self, step: int = 0):
        for name, values in self.batch_metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []

            # Add the average of metrics values in the batch.
            self.metrics[name].append((step, sum(values) / len(values)))

        self.batch_metrics.clear()

    def format(self, fstring: str) -> str:
        return fstring.format(**{
            k.replace('/', '_'): v[-1][1] for k, v in self.metrics.items()})