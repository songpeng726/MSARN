import argparse
import functools
import os
from datetime import datetime

from matplotlib import pyplot as plt
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
from  MultiRes1D import *



import numpy as np
import torch
from torch.utils.data import DataLoader
# from resnet import resnet34
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from reader import CustomDataset
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size',       int,    16,                       '训练的批量大小')
add_arg('num_workers',      int,    4,                        '读取数据的线程数量')
add_arg('num_epoch',        int,    101,                       '训练的轮数')
add_arg('num_classes',      int,    10,                       '分类的类别数量')
add_arg('learning_rate',    float,  1e-2,                     '初始学习率的大小')
add_arg('input_shape',      str,    '(None, 1, 80000)',    '数据输入的形状')
add_arg('train_list_path',  str,    'audio/train_list.txt', '训练数据的数据列表路径')
add_arg('test_list_path',   str,    'audio/test_list.txt',  '测试数据的数据列表路径')
add_arg('save_model',       str,    'models/',                '模型保存的路径')
add_arg('load_model',       str,    default='checkpoints/checkpoint_99_epoch.pkl', help='保存点')
# add_arg('load_model',       str,    default=None , help='保存点')


args = parser.parse_args()
device = torch.device("cuda")
checkpoint = torch.load(args.load_model)
model = MultiRes1D().to(device)
model.load_state_dict(checkpoint['model_state_dict'])

test_dataset = CustomDataset(args.test_list_path, model='test')
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

# 评估模型
def test(model, test_loader, device):
    Emotion_kinds = 10
    conf_matrix = torch.zeros(Emotion_kinds, Emotion_kinds)
    model.eval()
    accuracies = []
    for batch_id, (spec_mag, label) in enumerate(test_loader):
        spec_mag = spec_mag.to(device)
        label = label.to(device).long()
        output = model(spec_mag)

        conf_matrix = confusion_matrix(output, label, conf_matrix)
        conf_matrix = conf_matrix.cpu()
        # print(output)
        output = output.data.cpu().numpy()
        output = np.argmax(output, axis=1)
        # print("output=",output)
        label = label.data.cpu().numpy()
        # print("label=",label)
        acc = np.mean((output == label).astype(int))
        accuracies.append(acc.item())

    conf_matrix = np.array(conf_matrix.cpu())  # 将混淆矩阵从gpu转到cpu再转到np
    corrects = conf_matrix.diagonal(offset=0)  # 抽取对角线的每种分类的识别正确个数
    per_kinds = conf_matrix.sum(axis=1)  # 抽取每个分类数据总的测试条数

    print("混淆矩阵总元素个数：{0},测试集总个数:{1}".format(int(np.sum(conf_matrix)), int(np.sum(conf_matrix))))
    print(conf_matrix)
    print("转为百分比")
    # conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    # np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    # print(conf_matrix)

    # 获取每种Emotion的识别准确率
    print("每种情感总个数：", per_kinds)
    print("每种情感预测正确的个数：", corrects)
    print("每种情感的识别准确率为：{0}".format([rate * 100 for rate in corrects / per_kinds]))
    print("平均识别准确率为:", sum(corrects / per_kinds) * 10)

    # 绘制混淆矩阵
    Emotion = 10  # 这个数值是具体的分类数，大家可以自行修改
    labels = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot',
              'jackhammer', 'siren', 'street_music']  # 每种类别的标签

    # 显示数据
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)

    # 在图中标注数量/概率信息
    thresh = conf_matrix.max() / 2  # 数值颜色阈值，如果数值超过这个，就颜色加深。
    for x in range(Emotion_kinds):
        for y in range(Emotion_kinds):
            # 注意这里的matrix[y, x]不是matrix[x, y]
            info = int(conf_matrix[y, x])
            plt.text(x, y, info,
                     verticalalignment='center',
                     horizontalalignment='center',
                     color="white" if info > thresh else "black")

    plt.tight_layout()  # 保证图不重叠
    plt.yticks(range(Emotion_kinds), labels)
    plt.xticks(range(Emotion_kinds), labels, rotation=45)  # X轴字体倾斜45°
    plt.savefig('confusion_matrix.png', dpi = 600)
    plt.show()

    plt.close()



    return float(sum(accuracies) / len(accuracies))





if __name__ == '__main__':
    acc = test(model, test_loader, device)
    print('acc=', acc)
    # cunfusion()

