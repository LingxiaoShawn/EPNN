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

    def __init__(self, dim_out,  batchnorm=False, layernorm=False, pad_to_full_graph=True) -> None:
        super().__init__()
        self.dim_out = dim_out
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        self.pad_to_full_graph = pad_to_full_graph

        self.f1 = MonotonicNN(dim_out//2, hidden_features=2048, nonlinear=True, exp=True,  bandpass=True,  monotonic=True, relu=False)
        self.f2 = MonotonicNN(dim_out//2, hidden_features=2048, nonlinear=True, exp=True,  bandpass=False, monotonic=True, relu=False)
        self.g =  MonotonicNN(dim_out,    hidden_features=64,   nonlinear=True, exp=False, bandpass=False, monotonic=False, relu=True)

        self.linear = nn.Linear(2*dim_out, dim_out)
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
        f1 = self.f1(batched_eigval.unsqueeze(-1)) # b x m_max x dim_out//2
        f2 = self.f2(batched_eigval.unsqueeze(-1)) # b x m_max x dim_out//2
        f = torch.cat([f1, f2], dim=-1) # b x m_max x dim_out
        f[~mask_m] = 0.0 # b x m_max x dim_out

        # get g
        g = self.g(batched_eigvec.unsqueeze(-1)) # b x m_max x n_max x n_max x dim_out
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
        out = torch.cat([batch.edge_attr, out], dim=-1) # num_edges_in_batch x (dim_out + dim_previous)
        batch.edge_attr = self.linear(out) # num_edges_in_batch x dim_out
    
        return batch

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

            if self.relu:
                _h = torch.relu(_h)
            else:
                _h = torch.sigmoid(_h) # 0 to 1
                if self.bandpass:
                    _h = self.bandpass(_h)
                _h = 2.0 * (_h - 0.5) # -1 to 1

            _h = self.l3(_h) / self.hidden_features
            h = h + _h
        if self.exp:
            h = torch.exp(h)
        return h # original_shape x out_features
    
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






