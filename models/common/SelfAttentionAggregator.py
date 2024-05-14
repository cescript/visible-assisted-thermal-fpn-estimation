import torch
from torch import nn
import torch.nn.functional as funs

class SelfAttentionAggregator(nn.Module):
    def __init__(self, encoding_size, hidden_size):
        super(SelfAttentionAggregator, self).__init__()

        # single head attention module
        self.embed_size = hidden_size

        # create query, key and value vectors based on the head dimension
        self.queries = nn.Linear(encoding_size, self.embed_size)
        self.keys = nn.Linear(encoding_size, self.embed_size)
        self.values = nn.Linear(encoding_size, encoding_size)
    
    def forward(self, x):
        # get the size of each dimension
        agg_size, batch_size, channels, height, width = x.shape
        
        # flatten feature vector x to make output xf: [BATCH x AGGREGATED x ENCODED_SIZE]
        xf = x.transpose(0, 1).view(batch_size, agg_size, -1)

        # compute linear transformations for the values, keys, and queries
        values = self.values(xf)
        keys = self.keys(xf)
        queries = self.queries(xf)
    
        # calculate the attention scores using batch matrix multiplication
        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.embed_size ** (1 / 2))
    
        # apply softmax to get the attention weights
        attention = funs.softmax(attention_scores, dim=-1)
    
        # apply the attention weights to the values out: [BATCH x AGGREGATED x ENCODED_SIZE]
        out = torch.bmm(attention, values)
        
        # get the mean output (Aggregation)
        out = torch.mean(out, dim=1)
        
        # reshape the output
        out = out.view(batch_size, channels, height, width)
    
        # return the reshaped output
        return out
