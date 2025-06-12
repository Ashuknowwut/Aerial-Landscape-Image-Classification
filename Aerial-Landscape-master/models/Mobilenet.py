import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import os
import time
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import torch.nn.init as init
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--train_model', default='mobilenet-v2', type=str, help="trained model type")
    parser.add_argument('-p', '--load_pretrained', default=True, type=bool, help="whether load pretrained model")
    parser.add_argument('-da', '--data_aug', default=True, type=str2bool, help="whether perform data augmentation for training data")
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, help="learning rate")
    parser.add_argument('-mo', '--momentum', default=0.9, type=float, help="momentum")
    parser.add_argument('-e', '--epochs', default=50, type=int, help="epochs")
    parser.add_argument('-o', '--save_folder', default='saved_model', type=str, help="path for saved trained model")
    parser.add_argument('-bs', '--batch_size', default=64, type=int, help="batch size")
    return parser.parse_args()


class Train:
    use_gpu = torch.cuda.is_available()
    loss_func = nn.CrossEntropyLoss()

    def __init__(self, args):
        self.args = args
        self.net = None
        self.train_dataloader = None
        self.test_dataloader = None


    def create_model(self):
        # define model
        if 'mobilenet-v2' in self.args.train_model:
            '''
            load pre-constructed MobileNet-V2 model by Pytorch: https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
            source code: https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
            paper: https://arxiv.org/abs/1801.04381
            '''
            self.net = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', \
                                      pretrained=self.args.load_pretrained)
        else:
            pass

        if self.use_gpu:
            self.net.cuda()

        print("use gpu: ", self.use_gpu)
        print("model type: ", self.args.train_model)

        if  not self.args.load_pretrained:
            print("Init model weights")
            self.net.apply(self.init_weights)

        # optimizer
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.learning_rate, momentum=self.args.momentum)
        # self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.learning_rate)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[40], gamma=0.3)


    def build_dataloader(self):
        train_dir = './Aerial_Landscapes_split/train/'
        test_dir = './Aerial_Landscapes_split/test/'
        # define transformation
        if self.args.data_aug:
            train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        else:
            train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])

        train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
        test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

        # Create DataLoaders
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4
        )

        self.test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4
        )

    def training(self):
        epoch = 0
        metric_cur = -np.inf
        while epoch < self.args.epochs:
            self.net.train()
            self.scheduler.step()
            losses = []
            for i, (batch_x, batch_y) in enumerate(self.train_dataloader):
                if self.use_gpu:
                    data, target = Variable(batch_x).cuda(), Variable(batch_y).cuda()
                else:
                    data, target = Variable(batch_x), Variable(batch_y)

                self.optimizer.zero_grad()
                out = self.net(data)
                loss = self.loss_func(out, target)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

            print(f"Epoch: {epoch}")
            print(f"Train Loss: {np.mean(losses)}")
            metric = self.testing()
            # save the current best model
            if metric_cur < metric:
                self.save_model(cur_epoch=epoch)
                print('model saved')
                metric_cur = metric
            epoch += 1
        self.save_model(cur_epoch='last')
        print('last model saved')

    def testing(self):
        self.net.eval()
        y_true = []
        y_pred = []
        for i, (batch_x, batch_y) in enumerate(self.test_dataloader):
            if self.use_gpu:
                data = Variable(batch_x).cuda()
            else:
                data = Variable(batch_x)
            target = batch_y
            output = self.net(data)
            if self.use_gpu:
                output = output.cpu()
            predicted = output.argmax(dim=1, keepdim=False)
            y_true.append(target)
            y_pred.append(predicted)
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        # understand the differences among Accuracy, Precision, Recall and F1-score:
        # https://zhuanlan.zhihu.com/p/147663370
        # https://zhuanlan.zhihu.com/p/405658103
        accuracy = accuracy_score(y_true, y_pred)
        print('test accuracy: ', accuracy)
        precision = precision_score(y_true, y_pred, average='weighted')
        print('test precision: ', precision)
        recall = recall_score(y_true, y_pred, average='weighted')
        print('test recall: ', recall)
        f1 = f1_score(y_true, y_pred, average='weighted')
        print('test F1 score score: ', f1)
        return accuracy # save_metric



    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)

    def print_model_parm_nums(self):
        model = self.net
        total = sum([param.nelement() for param in model.parameters()])
        print(f' Number of params: {total / 1e6}M')

    def save_model(self, cur_epoch):
        if not os.path.exists(self.args.save_folder):
            os.mkdir(self.args.save_folder)
        torch.save(self.net.state_dict(), os.path.join(self.args.save_folder, self.args.train_model + '_params_epoch{}.pkl'.format(cur_epoch)))
        # torch.save(self.net, os.path.join(self.args.save_folder, self.args.train_model + '_epoch{}.pkl'.format(cur_epoch)))


if __name__ == "__main__":
    args = get_args()
    print(args)
    trainer = Train(args=args)
    trainer.build_dataloader()
    trainer.create_model()
    trainer.print_model_parm_nums()

    training_start = time.time()
    trainer.training()
    print(f"Training Time: {time.time() - training_start}")
    # testing_start = time.time()
    # trainer.testing()
    # print(f"Testing Time: {time.time() - testing_start}")

