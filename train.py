import torch
import torch.nn as nn
import torchvision
from vggnet import VGGNet
from pre_resnet import pytorch_resnet18
from load_cifar10 import train_data_loader
from load_cifar10 import test_data_loader
import os
import tensorboardX

# if not os.path.exists("log"):
#     os.mkdir("log")
#
# writer = tensorboardX.SummaryWriter("log")

# 判断是否存在GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epoch_num = 200
lr = 0.01

net = pytorch_resnet18().to(device)

# loss
loss_func = torch.nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# optimizer = torch.optim.SGD(net.parameters(),lr=lr, momentum=0.9,weight_decay=5e-4)

# 变长的学习率(5个epoch进行衰减)
schedular = torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.9)

batch_size = 64

step_n = 0
for epoch in range(epoch_num):
    print("epoch is: ", epoch)
    net.train()  #  train BN dropout

    for i, data in enumerate(train_data_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        loss = loss_func(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _,pred = torch.max(outputs.data, dim=1)

        correct = pred.eq(labels.data).sum()

        print("step ", i, "loss is ", loss.item(), "mini-batch correct is: ", 100.0 * correct.item() / batch_size)

        # writer.add_scalar("train loss", loss.item(), global_step=step_n)
        # writer.add_scalar("train correct",  100.0 * correct.item() / batch_size,global_step=step_n)
        # writer.add_image(inputs)
        # step_n += 1

    if not os.path.exists("models"):
        os.mkdir("models")

    torch.save(net.state_dict(), "models\\{}.pth".format(epoch+1))
    schedular.step()
    print("lr is ", optimizer.state_dict()["param_groups"][0]["lr"])

    loss_sum = 0
    correct_sum = 0
    for i, data in enumerate(train_data_loader):
        net.eval()
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        loss = loss_func(outputs, labels)

        _, pred = torch.max(outputs.data, dim=1)

        correct = pred.eq(labels.data).sum()
        correct = 100.0 * correct / batch_size

        loss_sum += (loss/len(test_data_loader))
        correct_sum += (correct/len(test_data_loader))

#         writer.add_scalar("test loss", loss, global_step=step_n)
#         writer.add_scalar("test correct",100.0 * correct.item() / batch_size, global_step=step_n)
#         writer.add_image(inputs)
#
#
# writer.close()