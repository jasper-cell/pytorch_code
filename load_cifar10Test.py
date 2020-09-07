from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from PIL import Image

label_name = ["airplane",
              "automobile",
              "bird",
              "cat",
              "deer",
              "dog",
              "frog",
              "horse",
              "ship",
              "truck"]


label_dict = {}

for idx, name in enumerate(label_name):
    label_dict[name] = idx

print(label_dict)

def default_loader(path):
    # 使用PIL的Image包的将图片路径打开后进行相应的加载，并利用convert方法将其颜色转换为RGB
    return Image.open(path).convert("RGB")

# 基于torchvision包下的transform进行变换的创建，采用的是Compose方法
train_transform = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(0.1),
    transforms.ToTensor()
])

class MyDataset(Dataset):
    def __init__(self, im_list, transforms=None, default_loader=default_loader):
        super(MyDataset, self).__init__()
        # 用一个列表进行相应的图像存储
        imgs = []

        for item in im_list:
            im_name = item.split("\\")[-2]
            im_label = label_dict[im_name]
            # 再以列表的形式对图片的地址及其所属类别进行打包编号
            imgs.append([item, im_label])

        # 将图像包、转换方式、数据加载器设置为属性进行类之中的存储，方便数据的传输
        self.imgs = imgs
        self.transfroms = transforms
        self.loader = default_loader

    # 为了能够对数据进行迭代必须采用这种__getitem__方法
    def __getitem__(self, index):
        im_path, im_label = self.imgs[index]

        im_data = self.loader(im_path)

        if self.transfroms != None:
            im_data = self.transfroms(im_data)

        return im_data, im_label

    # 同样是为了能够方便的进行迭代
    def __len__(self):
        return len(self.imgs)

import glob

train_list = glob.glob("E:\\imooc\\pytorch_course\\06\\Cifar10\\Train02\\*\\*.png")
test_list = glob.glob("E:\\imooc\\pytorch_course\\06\\Cifar10\\Test02\\*\\*.png")

train_dataset = MyDataset(train_list, transforms=train_transform)
test_dataset = MyDataset(test_list, transforms=transforms.ToTensor())

# dataset batchsize shnuffle
# torch.util.data 下的 Dataloader包
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=6,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=6,
                             shuffle=False)

print("train datasets length: " , len(train_dataset))
print("test dataset length: ", len(test_dataset))