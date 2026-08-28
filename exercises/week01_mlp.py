#拟合非线性函数
import random
import time

import numpy as np #常用的数值计算库
import torch
from torch import nn


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    def __init__(self):
        super().__init__() #super().__init__() 初始化父类 nn.Module 的内部功能
        self.network = nn.Sequential( #定义网络层，nn.Sequential 表示数据按顺序依次通过各层：
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


def train(optimizer_name: str):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate data for y = 2x^2 + 3x + 1. 从-2到2均匀取2000个点 
    x = torch.linspace(-2, 2, 2000).reshape(-1, 1).to(device)
    y = 2 * x**2 + 3 * x + 1

    model = MLP().to(device)
    loss_fn = nn.MSELoss()#nn.MESloss是一个基类，nn.MESloss()创建了这个类的一个对象loss_fn，这个loss_fn可以和函数一样调用，因为实现了__call__方法

    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    start = time.perf_counter()
    for epoch in range(1000):
        # Forward pass.
        
        prediction = model(x)
        loss = loss_fn(prediction, y)

        # Backward pass and parameter update.
        optimizer.zero_grad() #清空旧的梯度
        loss.backward()#反向传播就是求出梯度
        optimizer.step()#根据各自参数的梯度更新参数

        if epoch % 200 == 0:
            print(f"{optimizer_name} epoch={epoch:4d} loss={loss.item():.6f}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    with torch.no_grad():
        test_x = torch.tensor([[-1.0], [0.0], [1.0]], device=device)
        prediction = model(test_x).cpu().flatten()
        expected = 2 * test_x.cpu().flatten() ** 2 + 3 * test_x.cpu().flatten() + 1

    print(f"\n{optimizer_name} final result")
    print("device:", device)
    print("elapsed:", round(elapsed, 3), "seconds")
    print("prediction:", prediction.tolist())
    print("expected:", expected.tolist())
    print("final loss:", loss.item())
    print("-" * 50)

    assert loss.item() < 0.05
    return loss.item(), elapsed


if __name__ == "__main__":
    sgd_loss, sgd_time = train("SGD")
    adam_loss, adam_time = train("Adam")

    print("SGD loss:", sgd_loss)
    print("Adam loss:", adam_loss)
    print("SGD time:", sgd_time)
    print("Adam time:", adam_time)
    print("Two-layer MLP training passed")




