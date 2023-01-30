# from keras.models import Model
# from keras.utils import np_utils
import torchvision

# from  MultiRes1D import *
import matplotlib.pyplot as plt
from sklearn import manifold
import numpy as np
# from keras.callbacks import ReduceLROnPlateau
import matplotlib as mpl
from reader import CustomDataset
import argparse
import functools
from utility import add_arguments, print_arguments
from torch.utils.data import DataLoader
import torch

from  CBAMRes1D import *



def one_hot(x, class_count):
	# 第一构造一个[class_count, class_count]的对角线为1的向量
	# 第二保留label对应的行并返回
	return torch.eye(class_count)[x,:]


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size',       int,    16,                       '训练的批量大小')
add_arg('num_workers',      int,    4,                        '读取数据的线程数量')
add_arg('num_epoch',        int,    101,                       '训练的轮数')
add_arg('num_classes',      int,    10,                       '分类的类别数量')
add_arg('learning_rate',    float,  1e-2,                     '初始学习率的大小')
add_arg('input_shape',      str,    '(None, 1, 80000)',    '数据输入的形状')
add_arg('train_list_path',  str,    'audio/train_list.txt', '训练数据的数据列表路径')
add_arg('test_list_path',   str,    'audio/train_list.txt',  '测试数据的数据列表路径')
add_arg('save_model',       str,    'models/',                '模型保存的路径')
add_arg('load_model',       str,    default='checkpoints/M3/checkpoint_120_epoch.pkl', help='保存点')
# add_arg('load_model',       str,    default=None , help='保存点')

args = parser.parse_args()

train_dataset = CustomDataset(args.train_list_path, model='train')
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
test_dataset = CustomDataset(args.test_list_path, model='test')
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
checkpoint = torch.load(args.load_model)
device = torch.device("cuda")



n_neighbors = 10   #一共有多少个类别
n_components = 2  #降维成几维 2或者3


def TSNE(args):
    model = MultiRes1D().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # print("model=", model)  # 可以查看每一层的名字
    # new_m = torchvision.models._utils.IntermediateLayerGetter(model, {'maxpool2': 'feat1'})
    # x_2, x = model(torch.rand(16, 1, 176400))
    # print(x_2.shape)
    # print(x.shape)
    # x3 = new_m(x_2)
    # print([(k, v.shape) for k, v in x3.items()])
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.load_weights('alex.h5', by_name=True)
    # model.eval()
    X = torch.empty(16,10)
    Y = torch.empty(16,10)
    X2 = torch.empty(16, 3, 126, 1175)
    i = 0
    for batch_id, (spec_mag, label) in enumerate(test_loader):
        spec_mag = spec_mag.to(device)
        # label = label.long()
        label = label.to(device).long()
        x2, output = model(spec_mag)
        # ok = torch.nn.Softmax()
        # output = ok(output)
        output = output.data.cpu().numpy()
        x2 = x2.data.cpu().numpy()
        # output = np.argmax(output, axis=1)
        label = label.data.cpu().numpy()
        # x2 = x2.data.cpu().numpy()
        # print(x2.shape)
        # print(x2.shape)
        # print(label)
        # x2 = x2.detach().numpy()
        # output = output.detach().numpy()
        # print(spec_mag.shape, label.shape)
        y_train = one_hot(label, 10)  #转为one-hot编码
        color = y_train
        color = [np.argmax(i) for i in color]  # 将one-hot编码转换为整数
        color = np.stack(color, axis=0)
        # 创建自定义图像

        # 训练模型
        o = output.shape[0]
        out = output.reshape((o, -1))
        out = torch.from_numpy(out)
        color = torch.from_numpy(color)
        if i != 0:
            X = torch.cat((X, out), 0)
            Y = torch.cat((Y, color), 0)
        else:
            X = out
            Y = color
        i += 1
    print(X.shape)
    fig = plt.figure(figsize=(12, 12))  # 指定图像的宽和高

    # t-SNE的最终结果的降维与可视化
    ts = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
    y = ts.fit_transform(X)
    ax1 = fig.add_subplot(3, 1, 2)

    cm = 'tab10'  # 调整颜色
    # cm = colormap()
    plt.scatter(y[:, 0], y[:, 1], c=Y, cmap=cm)
    ax1.set_title('T-SNE Scatter Plot', fontsize=14)

        # 绘制S型曲线的3D图像
        # ax = fig.add_subplot(313, projection='3d')  # 创建子图
        # ax.scatter(y[:, 0], y[:, 1], y[:, 2], c=color, cmap=cm)  # 绘制散点图，为不同标签的点赋予不同的颜色
        # ax.set_title('Original S-Curve', fontsize=14)
        # ax.view_init(4, -72)  # 初始化视角
        #
        # # t-SNE对原始图像的降维与可视化
        # ts = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
        # # 训练模型
        # X_test = X_train.reshape((len(X_train), img_size * img_size * 3))
        # y = ts.fit_transform(X_test)
        # ax1 = fig.add_subplot(3, 1, 1)
        # plt.scatter(y[:, 0], y[:, 1], c=color, cmap=cm)
        # ax1.set_title('Raw Data Scatter Plot', fontsize=14)

        # 显示图像
    plt.savefig(fname=r"C:\Users\songpeng\Desktop\小论文绘图及参考文献\T-SNE-Urban2.png", dpi=600)
    plt.show()

        # print(y_train.shape)
        # lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
        # optimizer = torch.optim.SGD(params=model.parameters(), lr=args.learning_rate)





if __name__ == '__main__':
    TSNE(args)
