import torch
import torch.nn as nn
from attrdict import AttrDict
from transformers import BertModel, BertConfig


class BertClassifier(nn.Module):
    def __init__(self, args):
       super(BertClassifier, self).__init__()
       bert_config = BertConfig()

       bert_config.num_hidden_layers = args.bert_layers
       bert_config.hidden_dropout_prob = args.dropout
       bert_config.attention_probs_dropout_prob = args.attention_dropout
       bert_config.num_attention_heads = args.num_attention_heads
       bert_config.hidden_size = args.hidden_size
       bert_config.intermediate_size = args.intermediate_size
       self.config = bert_config

       #print(bert_config)
       self.bert = BertModel(bert_config)
       classes = 1
       self.input_dim = args.input_dim
       self.in_conv = nn.Conv1d(self.input_dim, bert_config.hidden_size, kernel_size=1)
       self.out_linear = nn.Linear(self.bert.config.hidden_size, classes*args.nway)
       #self.final_act = nn.Tanh()


    def forward(self, inputs_embeds):
       # inputs_embeds: (batch_size, seq_length, input_dim)
       out = torch.transpose(inputs_embeds, 2, 1)
       out = torch.transpose(self.in_conv(out), 2, 1) # (batch_size, seq_length, hidden_size)
       _, out = self.bert(inputs_embeds=out, return_dict=False)
       out = self.out_linear(out)
       #out = self.final_act(self.out_linear(out))
       #out = out + prev.view(prev.shape[0], -1)
       return out

'''
import yaml
with open('./data_config.yaml', 'r') as fp:
    args = AttrDict(yaml.load(fp, Loader=yaml.FullLoader))

device = 'cuda:5'
bert = Bert(args, 314).to(device)
inputs_embeds = torch.randn(1,5,314).to(device)
out = bert(inputs_embeds)
print(out)
'''
