import torch
from demo_reg import Net

# data
import numpy as np
import re

filename = r'housing.data'
ff = open(filename).readlines()
data = []
for item in ff:
    out = re.sub(r'\s{2,}', " ", item).strip()
    out = out.split(" ")
    data.append(out)

data = np.array(data, dtype=np.float)
print(data.shape)

X = data[:, 0:-1]
Y = data[:,-1]

X_train = X[0:496, ...]
Y_train = Y[0:496, ...]
X_test = X[496:, ...]
Y_test = Y[496:, ...]

net = torch.loads("./model/model.pkl")

loss_func = torch.nn.MSELoss()

x_data = torch.tensor(X_test, dtype=torch.float32)
y_data = torch.tensor(Y_test, dtype=torch.float32)
pred = net.forward(x_data)
pred = torch.squeeze(pred)
loss = loss_func(pred, y_data)*0.001
print(" loss_test:{}".format(loss))