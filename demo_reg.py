import torch

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

# net
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(Net, self).__init__()

        self.hidden = torch.nn.Linear(n_feature, 100)
        self.predict = torch.nn.Linear(100, n_output)

    def forward(self, x):
        out = self.hidden(x)
        out = torch.relu(out)
        out = self.predict(out)
        return out

net = Net(13, 1)

# loss
loss_func = torch.nn.MSELoss()

# optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# training
for i in range(10000):
    x_data = torch.tensor(X_train, dtype=torch.float32)
    y_data = torch.tensor(Y_train, dtype=torch.float32)
    pred = net.forward(x_data)
    pred = torch.squeeze(pred)
    loss = loss_func(pred, y_data) * 0.001
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i%1000 == 0:
        print("iter:{}, loss_train:{}".format(i, loss))
        print(y_data[0:10])
        print(pred[0:10])
    # test
    x_data = torch.tensor(X_test, dtype=torch.float32)
    y_data = torch.tensor(Y_test, dtype=torch.float32)
    pred = net.forward(x_data)
    pred = torch.squeeze(pred)
    loss = loss_func(pred, y_data) * 0.001
    if i%1000 == 0:
        print("iter:{}, loss_test:{}".format(i, loss))

# torch.save(net, "model/model.pkl")
torch.save(net.state_dict(), "params.pkl")
