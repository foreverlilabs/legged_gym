# 第一版手搓
# import torch
# # y = 3x + 2

# x = torch.tensor([1.,2.,3.,4.])
# y = torch.tensor([5.,8.,11.,14.])

# w = torch.tensor(1.0,requires_grad = True)
# b = torch.tensor(0.0,requires_grad = True)

# y_hat = w*x+b
# loss=((y_hat-y)**2).mean()
# loss.backward()
# print(w.grad)
# print(b.grad)

# learning_rate=0.01

# with torch.no_grad():
#     w -= learning_rate*w.grad
#     b -= learning_rate*b.grad
# print(w)
# print(b)

import torch
# y = 3*x^2 + 2x + 5

x = torch.tensor([1.,2.,3.,4.])
y = torch.tensor([10.,21.,38.,61.])

w = torch.tensor(1.0,requires_grad = True)
b = torch.tensor(0.0,requires_grad = True)
c = torch.tensor(0.0,requires_grad = True)

learning_rate = 0.01

for epoch in range(10000):
    y_hat = w*(x**2)+b*x+c
    loss = ((y_hat-y)**2).mean()
    loss.backward()
    with torch.no_grad():#这句torch.no_grad()很重要
        w-=learning_rate*w.grad
        b-=learning_rate*b.grad
        c-=learning_rate*c.grad

    w.grad.zero_()
    b.grad.zero_()
    c.grad.zero_()

print("w:",w)
print("b:",b)
print("c:",c)



#深度学习的核心循环流程
# for epoch in range(epochs):
#     #输入数据
#     y_hat = model(x)
#     #计算损失函数
#     loss = criterion(y_hat,y)
#     #梯度清零
#     optimizer.zero_grad()
#     #反向传播
#     loss.backward()
#     #优化模型
#     optimizer.step()
# Forward->loss->Backward->Update->grad.zero_()