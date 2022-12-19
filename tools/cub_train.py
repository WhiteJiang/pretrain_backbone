# -*- coding: utf-8 -*-
# @Time    : 2022/12/19
# @Author  : White Jiang
from dataset.read_data import Read_Dataset
import random
import torch.nn as nn
from model.backbone.resnet import ResNet
from dataset.dataset import *
from config import cfg
from utils.meter import AverageMeter
from loguru import logger
import numpy as np


def main():
    if not os.path.exists('logs/' + cfg.MODEL.INFO):
        os.makedirs('logs/' + cfg.MODEL.INFO)
    logger.add('logs/' + cfg.MODEL.INFO + '/' + 'train.log')
    device = cfg.MODEL.DEVICE
    train_data = CUB(cfg.DATASETS.ROOT_DIR, is_train=True)
    train_label = train_data.train_label
    num_train = len(train_label)

    train_dataloader, num_train, test_dataloader, num_test = Read_Dataset(cfg, num_train)

    def seed_torch(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    seed = 42
    seed_torch(seed)
    print(f'seed:{seed}')
    model = ResNet(class_num=200, dim=512)

    criterion = nn.CrossEntropyLoss()
    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=cfg.SOLVER.BASE_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)
    model = model.to(device)
    model = nn.DataParallel(model)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    best_acc = 0.0
    best_model_wts = model.state_dict()

    for epoch in range(1, cfg.SOLVER.MAX_EPOCHS + 1):
        cur_lr = scheduler.get_lr()
        model.train()
        loss_meter.reset()
        acc_meter.reset()
        # loss = 0.0
        # prec = 0.0
        for i, (index, img, label) in enumerate(train_dataloader):
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            raw_logits = model(img)
            raw_loss = criterion(raw_logits, label)
            raw_loss.backward()

            optimizer.step()
            loss_meter.update(raw_loss.item(), img.shape[0])
            pred = raw_logits.max(1, keepdim=True)[1]
            pred = pred.eq(label.view_as(pred)).sum().item() / img.shape[0]
            acc_meter.update(pred)
            if (i + 1) % 20 == 0:
                logger.info(
                    '[epoch:{}/{}]Iteration[{}/{}][loss:{:.3f}][acc:{:.3f}][lr:{}'.format(
                        epoch, cfg.SOLVER.MAX_EPOCHS, (i + 1), len(train_dataloader),
                        loss_meter.avg,
                        acc_meter.avg,
                        cur_lr))
        logger.info('[epoch:{}/{}][loss:{:.3f}][acc:{:.3f}][lr:{}'.format(
                epoch, cfg.SOLVER.MAX_EPOCHS,
                loss_meter.avg,
                acc_meter.avg,
                cur_lr))
        test_acc = 0.0
        model.eval()
        for n_iter, (index, img, label) in enumerate(test_dataloader):
            with torch.no_grad():
                img = img.to(device)
                label = label.to(device)
                raw_logits = model(img)
                pred = raw_logits.max(1, keepdim=True)[1]
                test_acc += pred.eq(label.view_as(pred)).sum().item()
        logger.info('best_acc{:.3f}'.format(best_acc))
        logger.info('test_acc{:.3f}'.format(test_acc / num_test))
        if test_acc / num_test > best_acc:
            best_acc = test_acc / num_test
            best_model_wts = model.state_dict()
        if epoch % cfg.SOLVER.MAX_EPOCHS == 0:
            # if torch.cuda.device_count() > 1:
            #     torch.save(model.module.state_dict(),
            #                os.path.join(output_dir, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            # else:
            model.load_state_dict(best_model_wts)
            torch.save(model.module.state_dict(),
                       os.path.join(cfg.MODEL.CHECKPOINTS, cfg.MODEL.NAME + '_{}_best.pth'.format(best_acc)))


if __name__ == '__main__':
    main()
