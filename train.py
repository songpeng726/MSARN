import argparse
import functools
import os
from datetime import datetime
from tqdm import tqdm

# import wandb
# wandb.init(project="my-test-project", entity="spovo")
from torch.nn import init

# from model3 import *
# from WaveMsNet import *
# from ResNet1D import *
# from  MultiModel2D import *
# from  MultiModel import *
# from  AttenMulti import *
# from  MultiRes1D import *
# from  M5 import *
from  M4 import *
# from  M2 import *
# from  MultiCNN1D import *
# from  CNN1D import *
# from  CBAMCNN import *
# from  AResNet1D import *
# from  ACNN1D import *
# from  S1 import *
# from  S2 import *
# from  S4 import *
# from  S1new import *
# from  CBAMRes6 import *
# from  CBAMRes5 import *
# from  CBAMRes4 import *
# from  CBAMRes3 import *
# from  CBAMRes6 import *
# from  CBAMRes1D import *
# from  S41 import *
# from  S201 import *
# from  S1001 import *


import numpy as np
import torch
from torch.utils.data import DataLoader
# from resnet import resnet34
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from reader import CustomDataset
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size',       int,    12,                       '训练的批量大小')
add_arg('num_workers',      int,    8,                        '读取数据的线程数量')
add_arg('num_epoch',        int,    61,                       '训练的轮数')
add_arg('num_classes',      int,    10,                       '分类的类别数量')
add_arg('learning_rate',    float,  1e-2,                     '初始学习率的大小')
add_arg('input_shape',      str,    '(None, 1, 80000)',    '数据输入的形状')
add_arg('train_list_path',  str,    'audio/train_list.txt', '训练数据的数据列表路径')
add_arg('test_list_path',   str,    'audio/test_list.txt',  '测试数据的数据列表路径')
add_arg('save_model',       str,    'models/',                '模型保存的路径')
# add_arg('load_model',       str,    default='checkpoints/checkpoint_20_epoch.pkl', help='保存点')
add_arg('load_model',       str,    default=None , help='保存点')

args = parser.parse_args()

# wandb.config = {
#   "learning_rate": 1e-3,
#   "epochs": 5,
#   "batch_size": 32
# }

device = torch.device("cuda")

# model = ResNet1D().to(device)
# model = WaveMsNet().to(device)
# model = Model().to(device)

model = MultiRes1D().to(device)

# 评估模型
def test(model, test_loader, device):
    model.eval()
    accuracies = []
    for batch_id, (spec_mag, label) in enumerate(test_loader):
        spec_mag = spec_mag.to(device)
        label = label.to(device).long()
        output = model(spec_mag)
        # print(output)
        output = output.data.cpu().numpy()
        output = np.argmax(output, axis=1)
        # print("output=",output)
        label = label.data.cpu().numpy()
        # print("label=",label)
        acc = np.mean((output == label).astype(int))
        accuracies.append(acc.item())

    return float(sum(accuracies) / len(accuracies))



def train(args):

    # 数据输入的形状
    input_shape = eval(args.input_shape)
    # print("inputshpe=",input_shape)
    # 获取数据
    train_dataset = CustomDataset(args.train_list_path, model='train')
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    test_dataset = CustomDataset(args.test_list_path, model='test')
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    # 获取模型
    print("model=", model)
    # model.to(device)
    total = sum([param.nelement() for param in model.parameters()])

    print("Number of parameter: %.2fM" % (total / 1e6))
    # 获取优化方法
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.learning_rate)

    # 获取学习率衰减函数
    scheduler = StepLR(optimizer, step_size=8, gamma=0.7, verbose=True)
    # scheduler = MultiStepLR(optimizer, milestones=[60, 120, 140], gamma=0.1)


    # 获取损失函数
    loss = torch.nn.CrossEntropyLoss()
    # weight_decay = 100.0  # 正则化参数
    # if weight_decay > 0:
    #     reg_loss = Regularization(model, weight_decay, p=2).to(device)
    # else:
    #     print("no regularization")
    # 开始训练
    start_epoch = -1
    if args.load_model is not None:
        print("Continuing training full model from checkpoint " + str(args.load_model))
        checkpoint = torch.load(args.load_model)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for epoch in range(start_epoch + 1, args.num_epoch):
        model.train()
        loss_sum = []
        accuracies = []
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        # for batch_id, (spec_mag, label) in enumerate(train_loader):
        for batch_id, (spec_mag, label) in loop:
            spec_mag = spec_mag.to(device)
            # print(spec_mag.shape)
            # print(label)
            label = label.to(device).long()
            output = model(spec_mag)
            # print(output.shape)
            # 计算损失值
            los = loss(output, label)
            # if weight_decay > 0:
            #     los = los + reg_loss(model)
            # los = los.item()
            # print("loss=", los)
            optimizer.zero_grad()
            los.backward()
            # for name, parms in model.named_parameters():
            #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
            #           ' -->grad_value:', parms.grad)
            optimizer.step()

            # 计算准确率
            output = torch.nn.functional.softmax(output)
            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            # print(output)
            label = label.data.cpu().numpy()
            # print(label)
            acc = np.mean((output == label).astype(int))
            accuracies.append(acc)
            loss_sum.append(los)
            loop.set_description(f'Epoch [{epoch}/{args.num_epoch}]')
            loop.set_postfix(loss=los.item(), acc=acc)
            # if batch_id % 100 == 0:
        print('[%s] Train epoch %d, batch: %d/%d, loss: %f, accuracy: %f' % (
            datetime.now(), epoch, batch_id, len(train_loader), sum(loss_sum) / len(loss_sum), sum(accuracies) / len(accuracies)))
        scheduler.step()
        #保存检查点
        if epoch % 1 == 0:
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch}
            path_checkpoint = "checkpoints/checkpoint_{}_epoch.pkl".format(epoch)
            torch.save(checkpoint, path_checkpoint)

        # 评估模型
        acc = test(model, test_loader, device)
        print('='*70)
        print('[%s] Test %d, accuracy: %f' % (datetime.now(), epoch, acc))
        print('='*70)
        model_path = os.path.join(args.save_model, '7-16M4.pth')
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        torch.jit.save(torch.jit.script(model), model_path)
        # wandb.log({"loss": loss})

        # Optional
        # wandb.watch(model)



if __name__ == '__main__':
    print_arguments(args)
    train(args)
