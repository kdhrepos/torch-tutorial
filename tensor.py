import torch
import numpy as np

# data = [[1, 2], [3, 4]]
# x_data = torch.tensor(data)

# np_array = np.array(data)
# x_np = torch.from_numpy(np_array)

# # tensor generation
# x_ones = torch.ones_like(x_data, dtype=torch.float)
# x_rand = torch.rand_like(x_data, dtype=torch.float)
# tensor = torch.one(4, 4)

# # make tensor more faster
# if torch.cuda.is_available():
#     print("cuda is available")
#     tensor = tensor.to("cuda")

# # indexing, slicing
tensor = torch.rand(4, 4)
# print(f"Tensor: {tensor}")
# print(f"First row: {tensor[0]}")
# print(f"First column: {tensor[:, 0]}")
# print(f"Last column: {tensor[..., -1]}")

# # concatenation
# print(f"Concatenated tensor: {torch.cat([tensor, tensor, tensor], dim=0)}")
# print(f"Concatenated tensor: {torch.cat([tensor, tensor, tensor], dim=1)}")

# arithmetic operations
# matmul
t1 = tensor @ tensor.T
t2 = tensor.matmul(tensor.T)
print(f"matmul 1: {t2}")

t3 = torch.rand_like(t1)
torch.matmul(tensor, tensor.T, out=t3)
print(f"matmul 2: {t3}")

# element-wise product
e1 = tensor * tensor
e2 = tensor.mul(tensor)
print(f"element-wise product 1: {e1}")
print(f"element-wise product 2: {e2}")

e3 = torch.rand_like(e1)
torch.mul(tensor, tensor, out=e3)
print(f"element-wise product 3: {e3}")

# aggregate
agg = tensor.sum()
print(f"aggerated: {agg}")

# in-place
print(f"original : {tensor}")
tensor.add_(5)
print(f"add 5: {tensor}")

# torch to numpy
t = torch.ones(5)
print(f"torch: {t}")

n = t.numpy()
print(f"numpy: {n}")

# if tensor is changed, numpy does same
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# numpy to torch
n = np.ones(5)
t = torch.from_numpy(n)

# if numpy is changed, torch does same
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
