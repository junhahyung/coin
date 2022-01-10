import torch
import torch.optim as optim
from models.bert import BertClassifier


def get_model(args):
    model = BertClassifier(args)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    return model, optimizer, model.config
