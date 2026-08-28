# TensorBoard Training Curves

After training, start TensorBoard from the `legged_gym` workspace:

```bash
tensorboard --logdir logs --port 6006
```

Keep that terminal running, then open this address in a browser:

```text
http://localhost:6006
```

You can view training metrics such as reward and loss curves there.
