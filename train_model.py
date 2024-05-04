from training import Trainer
from data import TokenizedCorpus
import argparse
from train import GPT2TrainingSpec,TrainConfig
def add_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser('train', help='train GPT-2 model')

    group = parser.add_argument_group('Corpus and vocabulary')
    group.add_argument('--train_corpus', required=True,
                       help='training corpus file path')
    group.add_argument('--eval_corpus', required=True,
                       help='evaluation corpus file path')
    group.add_argument('--vocab_path', required=True,
                       help='vocabulary file path')

    group = parser.add_argument_group('Model configurations')
    group.add_argument('--seq_len', default=64, type=int,
                       help='maximum sequence length')
    group.add_argument('--layers', default=12, type=int,
                       help='number of transformer layers')
    group.add_argument('--heads', default=16, type=int,
                       help='number of multi-heads in attention layer')
    group.add_argument('--dims', default=1024, type=int,
                       help='dimension of representation in each layer')
    group.add_argument('--rate', default=4, type=int,
                       help='increase rate of dimensionality in bottleneck')
    group.add_argument('--dropout', default=0.1, type=float,
                       help='probability that each element is dropped')

    group = parser.add_argument_group('Training and evaluation')
    group.add_argument('--batch_train', default=64, type=int,
                       help='number of training batch size')
    group.add_argument('--batch_eval', default=64, type=int,
                       help='number of evaluation batch size')
    group.add_argument('--base_lr', default=1e-4, type=float,
                       help='default learning rate')
    group.add_argument('--wd_rate', default=1e-2, type=float,
                       help='weight decay rate')

    group.add_argument('--total_steps', default=1000000, type=int,
                       help='number of total training steps')
    group.add_argument('--eval_steps', default=500, type=int,
                       help='period to evaluate model and record metrics')
    group.add_argument('--save_steps', default=1000, type=int,
                       help='period to save training state to checkpoint')

    group = parser.add_argument_group('Saving and restoring')
    group.add_argument('--save_model_path', default='model.pth',
                       help='save trained model weights to the file')
    group.add_argument('--save_checkpoint_path', default='checkpoint.pth',
                       help='save training state to the checkpoint file')
    group.add_argument('--from_checkpoint', default=None,
                       help='load last training state from checkpoint file')
    group.add_argument('--from_pretrained', default=None,
                       help='initialize parameters from pretrained model')

    group = parser.add_argument_group('Extensions')
    group.add_argument('--use_amp', action='store_true',
                       help='use automatic mixed-precision in training')
    group.add_argument('--use_grad_ckpt', action='store_true',
                       help='use gradient checkpointing in transformer layers')
    group.add_argument('--gpus', default=None, type=int,
                       help='number of gpu devices to use in training')

    parser.set_defaults(func=train_gpt2_model)

import argparse

def train_gpt2_model(args: argparse.Namespace):

    train_data = "/home/agv/Documents/telugu-gpt/te.txt"
    val_data = "/home/agv/Documents/telugu-gpt/tel00.txt"
    tokenizer = "/home/agv/Documents/telugu-gpt/tel-gpt/wordpiece-telugu/config.json"

    spec = GPT2TrainingSpec(
        train_corpus=args.train_corpus, eval_corpus=args.eval_corpus,
        tokenizer_path=args.tokenizer, seq_len=args.seq_len, layers=args.layers,
        heads=args.heads, dims=args.dims, rate=args.rate, dropout=args.dropout,
        base_lr=args.base_lr, wd_rate=args.wd_rate,
        total_steps=args.total_steps, use_grad_ckpt=args.use_grad_ckpt)
    config = TrainConfig(
        batch_train=args.batch_train, batch_eval=args.batch_eval,
        total_steps=args.total_steps, eval_steps=args.eval_steps,
        save_steps=args.save_steps, save_model_path=args.save_model_path,
        save_checkpoint_path=args.save_checkpoint_path,
        description='Train GPT-2 model',
        log_format='train/loss: {train_loss:.4f}, eval/loss: {eval_loss:.4f}',
        use_amp=args.use_amp, gpus=args.gpus)

    Trainer(spec, config).train(from_checkpoint=args.from_checkpoint,
                                from_pretrained=args.from_pretrained)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser = subparsers.add_parser('train', help='train GPT-2 model')

    group = parser.add_argument_group('Corpus and vocabulary')
    group.add_argument('--train_corpus', required=False,default= "/home/agv/Documents/telugu-gpt/te.txt",
                       help='training corpus file path')
    group.add_argument('--eval_corpus', required=False,default="/home/agv/Documents/telugu-gpt/tel00.txt",
                       help='evaluation corpus file path')
    group.add_argument('--tokenizer', required=False,default= "/home/agv/Documents/telugu-gpt/tel-gpt/wordpiece-telugu/config.json",
                       help='tokenizer file path')

    group = parser.add_argument_group('Model configurations')
    group.add_argument('--seq_len', default=64, type=int,
                       help='maximum sequence length')
    group.add_argument('--layers', default=12, type=int,
                       help='number of transformer layers')
    group.add_argument('--heads', default=16, type=int,
                       help='number of multi-heads in attention layer')
    group.add_argument('--dims', default=1024, type=int,
                       help='dimension of representation in each layer')
    group.add_argument('--rate', default=4, type=int,
                       help='increase rate of dimensionality in bottleneck')
    group.add_argument('--dropout', default=0.1, type=float,
                       help='probability that each element is dropped')

    group = parser.add_argument_group('Training and evaluation')
    group.add_argument('--batch_train', default=64, type=int,
                       help='number of training batch size')
    group.add_argument('--batch_eval', default=64, type=int,
                       help='number of evaluation batch size')
    group.add_argument('--base_lr', default=1e-4, type=float,
                       help='default learning rate')
    group.add_argument('--wd_rate', default=1e-2, type=float,
                       help='weight decay rate')

    group.add_argument('--total_steps', default=1000000, type=int,
                       help='number of total training steps')
    group.add_argument('--eval_steps', default=500, type=int,
                       help='period to evaluate model and record metrics')
    group.add_argument('--save_steps', default=1000, type=int,
                       help='period to save training state to checkpoint')

    group = parser.add_argument_group('Saving and restoring')
    group.add_argument('--save_model_path', default='model.pth',
                       help='save trained model weights to the file')
    group.add_argument('--save_checkpoint_path', default='checkpoint.pth',
                       help='save training state to the checkpoint file')
    group.add_argument('--from_checkpoint', default=None,
                       help='load last training state from checkpoint file')
    group.add_argument('--from_pretrained', default=None,
                       help='initialize parameters from pretrained model')

    group = parser.add_argument_group('Extensions')
    group.add_argument('--use_amp', action='store_true',
                       help='use automatic mixed-precision in training')
    group.add_argument('--use_grad_ckpt', action='store_true',
                       help='use gradient checkpointing in transformer layers')
    group.add_argument('--gpus', default=None, type=int,
                       help='number of gpu devices to use in training')
    
    args = parser.parse_args()  
    train_gpt2_model(args)