from tokenizers import Tokenizer, trainers, pre_tokenizers,models
import argparse
import json
import glob
import os
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files", help = "file for trainging tokenizer ",type=str,default="/home/agv/Documents/telugu-gpt/te_out.txt")
    parser.add_argument("--vocab_size", help = "vocab size",type=int,default=250000)
    parser.add_argument("--output", help = "path to save preprocessed file",default="wordpiece-telugu")
    args = parser.parse_args()


    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]"]
    trainer = trainers.WordPieceTrainer(vocab_size=args.vocab_size, special_tokens=special_tokens)

    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # files = glob.glob(f"{args.input_files}/*")
    files = [args.input_files]
    tokenizer.train(files, trainer)

    os.mkdir(args.output)
    tokenizer.save(os.path.join(args.output,"config.json"))
    with open(f"{args.output}/config.json", 'r') as f:
        data = json.load(f)
        vocab = list(data['model']['vocab'].keys())

    with open(f"{args.output}/vocab.txt", 'w') as f:
        for word in vocab:
            f.write(word + '\n')