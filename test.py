import torch
t = torch.tensor([1, 2, 1])   # shape [2]

# make it [n,1] and [1,n]
diffs = t.unsqueeze(1) - t.unsqueeze(0)
print(diffs)