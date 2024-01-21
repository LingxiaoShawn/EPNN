import torch 
import torch.nn as nn
from torch_geometric.graphgym.register import register_edge_encoder
from grit.transform.eigenbasis import to_dense_grouped_EVD


@register_edge_encoder('eigenbasis_encoder')
class EigenBasisEncoder(torch.nn.Module):
    """
    For any graph, encode the eigenbasis to fixed length vector. 
    ---
    batched_eigen_value:  b x m_max, 
    batched_eigen_vector: b x m_max x n_max x n_max
    out: b x n_max x n_max x dim_out
    ---
    Formulation: 
    sum_m f(eigen_value_m) * g(eigen_vector_m), we assume f and g are monotonic scale (both input and output) functions,
    furthermore, we can allow dss based encoding in f and g, such that a summation is used to augment the transformation. 
    """

    def __init__(self, dim_out, eigval_hidden=256, eigspace_hidden=64, batchnorm=False, layernorm=False, pad_to_full_graph=True) -> None:
        super().__init__()
        self.dim_out = dim_out
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        self.pad_to_full_graph = pad_to_full_graph

        self.concat = False
        self.f = MonotonicNN(dim_out, hidden_features=eigval_hidden,    exp=True,  monotonic=False)
        self.g = MonotonicNN(dim_out, hidden_features=eigspace_hidden,  exp=False, monotonic=False)

        self.linear = nn.Linear(dim_out * (1+int(self.concat)), dim_out)
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(dim_out)
        if self.layernorm:
            self.ln = nn.LayerNorm(dim_out)

    def forward(self, batch):
        # assert batch contains necessary attributes
        assert hasattr(batch, 'grouped_eigval'), 'batch does not contain grouped_eigval'
        assert hasattr(batch, 'grouped_eigvec'), 'batch does not contain grouped_eigvec'
        assert hasattr(batch, 'group_batch'), 'batch does not contain group_batch'

        # get batched eigenvalues and eigenvectors 
        (
            batched_eigval, # b x m_max
            batched_eigvec, # b x m_max x n_max x n_max
            mask_m,         # b x m_max
            mask_all,       # b x m_max x n_max x n_max
        ) = to_dense_grouped_EVD(batch.grouped_eigval, batch.grouped_eigvec, batch.batch, batch.group_batch)

        # get f1 and f2, and concatenate them
        f = self.f(batched_eigval.unsqueeze(-1)) # b x m_max x dim_out
        f = f * mask_m.unsqueeze(-1) # b x m_max x dim_out

        # f2 = self.f2(batched_eigval.unsqueeze(-1)) # b x m_max x dim_out//2
        # f = torch.cat([f1, f2], dim=-1) # b x m_max x dim_out
        # f[~mask_m] = 0.0 # b x m_max x dim_out
        
        # get g
        g = self.g(batched_eigvec.unsqueeze(-1)) # b x m_max x n_max x n_max x dim_out 
        #### TODO: problem here, g cost too much memory by storage n^3 things. Solution: truncation is necessary.

        g[~mask_all] = 0.0 # b x m_max x n_max x n_max x dim_out

        # get the final output
        out = (f.unsqueeze(-2).unsqueeze(-2) * g).sum(dim=1) # b x n_max x n_max x dim_out

        # flatten the output to num_edges_in_batch x dim_out
        # notice that this assume pad to full graph is used 
        assert self.pad_to_full_graph, 'flatten output assumes pad_to_full_graph is used'
        out = out[mask_all[:,0]] # num_edges_in_batch x dim_out

        if self.batchnorm:
            out = self.bn(out)
        if self.layernorm:
            out = self.ln(out)

        # return out
        if self.concat:
            batch.edge_attr = self.linear(torch.cat([batch.edge_attr, out], dim=-1)) # num_edges_in_batch x dim_out
        else:
            batch.edge_attr = batch.edge_attr + self.linear(out)
        return batch

    
import torch.nn.functional as F
class MonotonicNN(nn.Module):
    def __init__(self, out_features, hidden_features=2048, nonlinear=True, exp=False, bandpass=False, monotonic=True, relu=False):
        super().__init__()
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.nonlinear = nonlinear
        self.bandpass = bandpass
        self.exp = exp
        self.relu = relu

        self.l1 = DenseMonotone(1, out_features, bias=True, monotonic=monotonic)
        self.l1.bias.data.fill_(0)
        if self.nonlinear:
            self.l2 = DenseMonotone(1, hidden_features, bias=True, monotonic=monotonic)
            self.l3 = DenseMonotone(hidden_features, out_features, bias=False, monotonic=monotonic)
            if bandpass:
                self.bandpass = GuassianBandPass(hidden_features)

    def forward(self, x):
        # x: b x n x ... x 1, elements in range [-1,1]
        # outputs: b x n x ... x out_features
        h = self.l1(x)
        if self.nonlinear:
            _h = self.l2(x)
            _h = F.silu(_h)
            _h = self.l3(_h) #/ self.hidden_features
            h = h + _h
        if self.exp:
            h = torch.exp(h)
        else:
            h = F.silu(h)
        return h # original_shape x out_features
    
class DenseMonotone(nn.Module):
    """Strictly increasing Dense layer in PyTorch."""

    def __init__(self, in_features, out_features, bias=True, monotonic=True):
        super().__init__()
        self.use_bias = bias
        self.monotonic = monotonic

        self.weight = nn.Parameter(torch.rand(in_features, out_features))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.use_bias:
            self.bias = nn.Parameter(torch.rand(out_features))
            nn.init.normal_(self.bias, mean=0.0, std=0.02)

    def forward(self, inputs):
        # inputs: b x n x ... x in_features
        # Ensure kernel is non-negative to maintain monotonicity
        if self.monotonic:
            kernel = torch.abs(self.weight)
        else:
            kernel = self.weight

        # Perform the matrix multiplication
        y = torch.matmul(inputs, kernel)

        # Add the bias if it's used
        if self.use_bias:
            y = y + self.bias
        return y
    
import math
class GuassianBandPass(nn.Module):
    # kind of band-pass filter
    def __init__(self, dim):
        super().__init__()
        self.dim = dim 
        self.mean = nn.Parameter(torch.rand(dim))
        self.std = nn.Parameter(torch.rand(dim))

    def forward(self, x):
        # x: b x n x ... x dim, assume x > 0 
        mean = (2.0*torch.sigmoid(self.mean)-1.0).view(*[1]*(x.dim()-1), -1)  ## we assume mean between -1 to 1 
        std = torch.abs(self.std).view(*[1]*(x.dim()-1), -1) 
        scale = torch.exp(-0.5 * ((x - mean) / std)**2) / (std * math.sqrt(2.0 * math.pi))
        return scale * x 






