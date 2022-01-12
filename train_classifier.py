import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from attrdict import AttrDict
from trainer_classifier import Trainer
from dataset import get_dataset, CoinDataset
from models.get_model import get_model


# main loop
def run(conf):
    args = conf

    # prepare argument

    device = 'cuda:0'
    models = []
    optimizers = []

    train_loader, valid_loader = get_dataset(args)
    for data in train_loader:
        args.input_dim = data[0].shape[-1]

    # number of ensemble
    for i in range(args.n_ensemble):
        model, optimizer, bert_config = get_model(args)
        models.append(model)
        optimizers.append(optimizer)

    print(bert_config)


    # prepare dataset
    # prepare loss function
    loss_fn = nn.CrossEntropyLoss()

    # load trainer
    trainer = Trainer(args,
                    models,
                    optimizers,
                    train_loader,
                    valid_loader,
                    loss_fn,
                    device)

    # start training
    trainer.train()

    print(f'best acc: {trainer.best_acc}')
    print(f'best confusion: {trainer.best_confusion}')
    

def add_global_args():
    args = argparse.ArgumentParser()
    args.add_argument("--pair", nargs="+", type=str, required=True, choices=['BTC', 'ETH'])
    args.add_argument("--intv", default="1h", type=str)
    args.add_argument("--nhist", default=100, type=int)
    args.add_argument("--ntarget", default=4, type=int)
    args.add_argument("--ratio", default=0.9, type=float)
    args.add_argument("--bs", default=2048, type=int)

    # model config
    args.add_argument("--n_ensemble", default=5, type=int)
    args.add_argument("--nway", default=2, type=int)
    args.add_argument("--n_epochs", default=500, type=int)
    args.add_argument("--save_freq", default=1000, type=int)
    args.add_argument("--val_freq", default=50, type=int)
    args.add_argument("--lr", default=0.0001, type=float)
    args.add_argument("--dropout", default=0.1, type=float)
    args.add_argument("--attention_dropout", default=0.1, type=float)
    args.add_argument("--num_attention_heads", default=6, type=int)
    args.add_argument("--hidden_size", default=192, type=int)
    args.add_argument("--intermediate_size", default=768, type=int)
    args.add_argument("--bert_layers", default=2, type=int)

    args.add_argument("--name", default='default', type=str)
    args.add_argument("--output_dir", default='/home/nas1_temp/junhahyung/trading/output/', type=str)
    
    # For local run
    args.add_argument('-f', type=str, default="")
    
    conf = args.parse_args()

    return conf


conf = add_global_args()
run(conf)
