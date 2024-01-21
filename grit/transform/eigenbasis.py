import torch
from torch_sparse import SparseTensor
from torch_geometric.utils import get_laplacian, to_undirected
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

class GroupedEVDTransform(object):
    def __init__(self, max_num_groups=20, threshold=1e-6,):
        self.threshold = threshold
        self.max_num_groups = max_num_groups
    def __call__(self, data):
        D, V = EVD_normlized_adj(data, norm='sym')
        # Group eigen vals and vecs 
        grouped_eigval = []
        grouped_eigvec = []
        preval = 1 # assume eigenvalues start from 1
        stack = []
        for val, vec in zip(D, V.T):
            if abs(val - preval) < self.threshold:
                stack.append(vec)
            else:
                grouped_eigval.append(preval)
                v = torch.stack(stack)
                grouped_eigvec.append(v.T.matmul(v))
                preval = val 
                stack = [vec]

        grouped_eigval.append(preval)
        v = torch.stack(stack)
        grouped_eigvec.append(v.T.matmul(v))
        grouped_eigval = torch.tensor(grouped_eigval) # m x 1 
        grouped_eigvec = torch.stack(grouped_eigvec)  # m x n x n 

        m = min(grouped_eigval.size(0), self.max_num_groups) # truncate the num of groups for scalability. 
        data.grouped_eigval = grouped_eigval[:m]
        data.grouped_eigvec = grouped_eigvec[:m].reshape(-1) # reshape to 1-d to save
        data.group_batch = torch.zeros(m, dtype=torch.long)
        return data
    

def EVD_normlized_adj(data, norm=None, topk=0):
    L_raw = get_laplacian(to_undirected(data.edge_index, num_nodes=data.num_nodes), 
                          normalization=norm, num_nodes=data.num_nodes)
    if topk > 0:
        try:
            L = L_raw[0].numpy(), L_raw[1].numpy()
            L = coo_matrix((L[1], (L[0][0], L[0][1])))
            D, V = eigsh(L, topk+1, which='SM')
        except:
            L = SparseTensor(row=L_raw[0][0], col=L_raw[0][1], value=L_raw[1], sparse_sizes=(data.num_nodes, data.num_nodes)).to_dense()
            D, V  = torch.linalg.eigh(L)
    else:
        L = SparseTensor(row=L_raw[0][0], col=L_raw[0][1], value=L_raw[1], sparse_sizes=(data.num_nodes, data.num_nodes)).to_dense()
        D, V  = torch.linalg.eigh(L)  
    D = 1 - D # in the range of -1 to 1, from large to small (1 to -1), this value can be related to tanh.
    return D, V


from torch_scatter import scatter_add
def to_dense_grouped_EVD(eigS_flatten, eigV_flatten, batch, group_batch):
    device = eigS_flatten.device
    batch_size = int(batch.max()) + 1
    # get m 
    num_eigvals = scatter_add(torch.ones_like(group_batch, device=device), group_batch, dim=0, dim_size=batch_size) # b x 1
    # get n 
    num_nodes = scatter_add(torch.ones_like(batch, device=device), batch, dim=0, dim_size=batch_size) # b x 1
    # get n_max and m_max
    n_max = int(num_nodes.max())  
    m_max = int(num_eigvals.max()) 

    # get batched eigenvalues: b x m_max
    eigS_batched = torch.zeros(batch_size, m_max, device=device)
    mask_m = num_eigvals.unsqueeze(1) > torch.arange(m_max, device=device).unsqueeze(0) # b x m_max
    eigS_batched[mask_m] = eigS_flatten

    # get batched eigenvectors: b x m_max x n_max x n_max
    eigV_batched = torch.zeros(batch_size, m_max, n_max, n_max, device=device)
    mask_n = num_nodes.unsqueeze(1) > torch.arange(n_max, device=device).unsqueeze(0)   # b x n_max
    mask_all = mask_n.unsqueeze(2) & mask_n.unsqueeze(1) # b x n_max x n_max
    mask_all = mask_all.unsqueeze(1) & mask_m.unsqueeze(-1).unsqueeze(-1) # b x m_max x n_max x n_max
    eigV_batched[mask_all] = eigV_flatten
    return eigS_batched, eigV_batched, mask_m, mask_all   # b x m_max, b x m_max x n_max x n_max, [b x m_max, b x m_max x n_max x n_max]
