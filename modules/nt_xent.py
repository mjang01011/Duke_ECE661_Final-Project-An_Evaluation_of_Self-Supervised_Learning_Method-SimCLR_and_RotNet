import torch
import torch.nn as nn
import torch.nn.functional as F
## Criterion for SimCLR
class NT_Xent(nn.Module):
    def __init__(self, temperature):
        super(NT_Xent, self).__init__()
        self.temperature = temperature

    def forward(self, z):
        # z is (2N, ___)
        # two_N = z.size(dim=0)
        # N = two_N // 2
        # s = torch.zeros([two_N, two_N], dtype=torch.float32)
        # for i in range(two_N): # loop {1...2N}
        #     for j in range(two_N): # loop {1...2N}
        #         z_i = z[i]; z_j = z[j]
        #         s[i][j] = self.similarity(z_i, z_j)

        # numerator = torch.exp(torch.div(s, self.temperature))
        # sum_term = numerator
        # diag_term = torch.diag(sum_term, 0)
        # # sum_term = sum_term.fill_diagonal_(0.) # remove all diagonal similarities
        # denominator = torch.sum(sum_term, dim=1) - diag_term
        # loss = -1 * torch.log(torch.div(numerator, denominator))

        # loss_val = 0
        # for k in range(N):
        #     loss_val += loss[2*k][2*k+1] + loss[2*k][2*k-1]
        
        # loss_val = loss_val / two_N

        ### https://github.com/dhruvbird/ml-notebooks/blob/main/nt-xent-loss/NT-Xent%20Loss.ipynb
        sim = F.cosine_similarity(z[None, :, :], z[:,None,:], dim=-1) 
        sim[torch.eye(sim.size(0)).bool()] = float("-inf")
        target = torch.arange(z.size(dim=0)) # size equal to Batch size
        target[0::2] += 1
        target[1::2] -= 1
        loss_val = F.cross_entropy(sim / self.temperature, target, reduction="mean")
        return loss_val
    