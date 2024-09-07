import torch

# z = Wx + b
x = torch.rand(5)
y = torch.rand(3)  # ground truth

# can get gradient by "requires_grad=True"
W = torch.rand(5, 3, requires_grad=True)
b = torch.rand(3, requires_grad=True)
z = torch.matmul(x, W) + b  # predicted
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# gradient
loss.backward()  # do back prop
print(f"Weight gradient: {W.grad}")  # now W has its own gradient
print(f"Weight gradient: {b.grad}")  # b also does

# we don't need gradient -> get more performance
z = torch.matmul(x, W)+b
print(z.requires_grad)  # True

with torch.no_grad():
    z = torch.matmul(x, W) + b
print(z.requires_grad)  # False

# can get the same result with "detach"
z = torch.matmul(x, W)+b
z_det = z.detach()
print(z_det.requires_grad)
