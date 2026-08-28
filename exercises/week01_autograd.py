import torch

# x = torch.tensor(2.0,requires_grad=True)

# y = x**2+3*x+1

# y.backward()

# print(f"x = {x.item()}")
# print(f"y = {y.item()}")
# print(f"dy/dx = {x.grad.item()}")

x = torch.tensor(2.0,requires_grad=True)

y1 = x**2
y1.backward()
print("第一次梯度：",x.grad.item())

y2 = 3*x
y2.backward()
print("第二次反向传播后的梯度",x.grad.item())

#常见问题：
# requires_grad=True有什么作用？
# y.backward()做了什么？
# 为什么连续调用两次backward()后梯度会累积成7？
# 为什么训练循环中通常需要optimizer.zero_grad()？

#答案：requires_grad=True 表示 PyTorch 要追踪 x 参与的运算，建立计算图，以便之后自动计算损失对 x 的导数。反向传播得到的梯度保存在
#   x.grad。
#   y.backward() 会从 y 沿计算图反向应用链式法则，计算 dy/dx，并把结果累加到叶子张量 x.grad。
#   此例中 x = 2：
#   - y1 = x**2，导数为 2x = 4，第一次调用后 x.grad = 4
#   - y2 = 3*x，导数为 3，第二次调用把它加到已有梯度：4 + 3 = 7
#   所以不是第二次的梯度本身是 7，而是 PyTorch 默认执行梯度累积。这对一个参数被多个损失项或多个计算路径共同使用时很有用。

#   训练循环通常在每次参数更新前调用 optimizer.zero_grad()，将上一次迭代残留在各参数 .grad 中的梯度清零。否则新 batch 的梯度会继续
#   叠加到旧 batch，导致更新方向和步长不再是当前 batch 损失对应的梯度。

# 总结：PyTorch 默认执行梯度累积

