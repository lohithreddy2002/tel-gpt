from transformers import PreTrainedTokenizerFast
import random
def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab[item])
    return output
def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
   
        if total_length <= max_num_tokens:
            break
        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()



# fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="/home/agv/Documents/telugu-gpt/bert-v2/config.json")
all_documents = [[]]
count = 0
# rng = random.Random(12345)
# with open("/home/agv/Documents/telugu-gpt/te.txt", 'r') as f:
#     for line in f:
#         line = line.strip()
#         if not line:
#             all_documents.append([])
#         tokens = fast_tokenizer.convert_ids_to_tokens(fast_tokenizer.encode(line))
#         if tokens:
#             all_documents[-1].append(tokens)
#         count += 1
#         if count % 1000000 == 0:
#             print(count)
#         if count == 100: 
#             break
# all_documents.append([])
# all_documents = [x for x in all_documents if x]
# rng.shuffle(all_documents)
# max_seq_length = 512
# do_whole_word_mask = True
# max_num_tokens = max_seq_length-3 
# target_seq_length = max_num_tokens
# short_seq_prob=0.1
# masked_lm_prob = 0.15
# if rng.random() < short_seq_prob:
#     target_seq_length = rng.randint(2, max_num_tokens)
# doc = all_documents[-1]
# i= 0
# chun = []
# current_length = 0
# while i <len(doc):
#     seg = doc[i]
#     chun.append(seg)
#     tokens_a = []
#     current_length += len(seg)
#     for j in range(len(chun)):
#         tokens_a.extend(chun[j])
#     if i == len(doc) -1 or current_length >= target_seq_length:
#         if chun:
#             truncate_seq_pair(tokens_a, [], max_num_tokens, rng)
#             tokens = []
#             segment_ids = []
#             tokens.append("[CLS]")
#             segment_ids.append(0)
#             for token in tokens_a:
#                 tokens.append(token)
#                 segment_ids.append(0)

#             tokens.append("[SEP]")
#             segment_ids.append(0)
#             cand_indexes = []
#             for i,token in enumerate(tokens):
#                 if token == "[CLS]" or token == "[SEP]":
#                     continue

#                 if (do_whole_word_mask and len(cand_indexes) >= 1 and token.startswith("##")):
#                     cand_indexes[-1].append(i)
#                 else:
#                     cand_indexes.append([i])
#             rng.shuffle(cand_indexes)
#             print(cand_indexes)

# class CfgNode:
#     """ a lightweight configuration class inspired by yacs """
#     # TODO: convert to subclass from a dict like in yacs?
#     # TODO: implement freezing to prevent shooting of own foot
#     # TODO: additional existence/override checks when reading/writing params?

#     def __init__(self, **kwargs):
#         self.__dict__.update(kwargs)

#     def __str__(self):
#         return self._str_helper(0)

#     def _str_helper(self, indent):
#         """ need to have a helper to support nested indentation for pretty printing """
#         parts = []
#         for k, v in self.__dict__.items():
#             if isinstance(v, CfgNode):
#                 parts.append("%s:\n" % k)
#                 parts.append(v._str_helper(indent + 1))
#             else:
#                 parts.append("%s: %s\n" % (k, v))
#         parts = [' ' * (indent * 4) + p for p in parts]
#         return "".join(parts)

#     def to_dict(self):
#         """ return a dict representation of the config """
#         return { k: v.to_dict() if isinstance(v, CfgNode) else v for k, v in self.__dict__.items() }

#     def merge_from_dict(self, d):
#         self.__dict__.update(d)

#     def merge_from_args(self, args):
#         """
#         update the configuration from a list of strings that is expected
#         to come from the command line, i.e. sys.argv[1:].

#         The arguments are expected to be in the form of `--arg=value`, and
#         the arg can use . to denote nested sub-attributes. Example:

#         --model.n_layer=10 --trainer.batch_size=32
#         """
#         for arg in args:

#             keyval = arg.split('=')
#             assert len(keyval) == 2, "expecting each override arg to be of form --arg=value, got %s" % arg
#             key, val = keyval # unpack

#             # first translate val into a python object
#             try:
#                 val = literal_eval(val)
#                 """
#                 need some explanation here.
#                 - if val is simply a string, literal_eval will throw a ValueError
#                 - if val represents a thing (like an 3, 3.14, [1,2,3], False, None, etc.) it will get created
#                 """
#             except ValueError:
#                 pass

#             # find the appropriate object to insert the attribute into
#             assert key[:2] == '--'
#             key = key[2:] # strip the '--'
#             keys = key.split('.')
#             obj = self
#             for k in keys[:-1]:
#                 obj = getattr(obj, k)
#             leaf_key = keys[-1]

#             # ensure that this attribute exists
#             assert hasattr(obj, leaf_key), f"{key} is not an attribute that exists in the config"

#             # overwrite the attribute
#             print("command line overwriting config attribute %s with %s" % (key, val))
#             setattr(obj, leaf_key, val)

from torch.utils.data import Dataset
import torch


# def get_config():

#     C = CfgNode()

#     # system
#     C.system = CfgNode()
#     C.system.seed = 3407
#     C.system.work_dir = './out/chargpt'

#     # data
#     C.data = TleuguDatset.get_default_config()

#     # model
#     # C.model = GPT.get_default_config()
#     C.model.model_type = 'gpt-mini'

#     # trainer
#     # C.trainer = Trainer.get_default_config()
#     C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster

#     return C


# class TleuguDatset(Dataset):
#     """
#         Emits batches of characters
#     """

#     @staticmethod
#     def get_default_config():
#         C = CfgNode()
#         C.block_size = 128
#         return C

#     def __init__(self, config, data):
#         self.config = config
#         self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=config.tokeizer_path)
#         chars = sorted(list(set(data)))
#         data_size, vocab_size = len(data), len(chars)
#         print('data has %d characters, %d unique.' % (data_size, vocab_size))

#         self.stoi = { ch:i for i,ch in enumerate(chars) }
#         self.itos = { i:ch for i,ch in enumerate(chars) }
#         self.vocab_size = vocab_size
#         self.data = data

#     def get_vocab_size(self):
#         return self.vocab_size

#     def get_block_size(self):
#         return self.config.block_size

#     def __len__(self):
#         return len(self.data) - self.config.block_size

#     def __getitem__(self, idx):
#         # grab a chunk of (block_size + 1) characters from the data
#         chunk = 
#         chunk = self.data[idx:idx + self.config.block_size + 1]
#         # encode every character to an integer
#         dix = [self.stoi[s] for s in chunk]
#         # return as tensors
#         x = torch.tensor(dix[:-1], dtype=torch.long)
#         y = torch.tensor(dix[1:], dtype=torch.long)
#         return x, y

# def load_dataset():
#     text = open('input.txt', 'r').read()
#     train_dataset = TleuguDatset(text,fast_tokenizer)
    


da = tokcor.fetch(8)
# fast_tokenizer.bos

print(fast_tokenizer.convert_ids_to_tokens(da["input"][0]))
print(fast_tokenizer.convert_ids_to_tokens(da["output"][0]))
