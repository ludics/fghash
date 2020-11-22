import torch
import torch.optim as optim
import os
import time
import models.alexnet as alexnet
import utils.evaluate as evaluate
import models.resnet as resnet


from loguru import logger
from models.adsh_loss import ADSH_Loss
from data.data_loader import sample_dataloader
from utils import AverageMeter
from models.network import AlexNet, ResNet
from utils.tools import compute_result, CalcTopMap

import numpy as np

class ArcLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(ArcLoss, self).__init__()
        self.gamma = config.gamma
        self.lambda1 = config.lambd
        self.K = bit
        self.one = torch.ones((config.batch_size, bit)).to(config.device)

    def d(self, hi, hj):
        inner_product = hi @ hj.t()
        norm = hi.pow(2).sum(dim=1, keepdim=True).pow(0.5) @ hj.pow(2).sum(dim=1, keepdim=True).pow(0.5).t()
        cos = inner_product / norm.clamp(min=0.0001)
        # formula 6
        return (1 - cos.clamp(max=0.99)) * self.K / 2

    def forward(self, u, y, ind):
        s = (y @ y.t() > 0).float()

        if (1 - s).sum() != 0 and s.sum() != 0:
            # formula 2
            positive_w = s * s.numel() / s.sum()
            negative_w = (1 - s) * s.numel() / (1 - s).sum()
            w = positive_w + negative_w
        else:
            # maybe |S1|==0 or |S2|==0
            w = 1

        d_hi_hj = self.d(u, u)
        # formula 8
        cauchy_loss = w * (s * torch.log(d_hi_hj / self.gamma) + torch.log(1 + self.gamma / d_hi_hj))
        # formula 9
        quantization_loss = torch.log(1 + self.d(u.abs(), self.one) / self.gamma)
        # formula 7
        loss = cauchy_loss.mean() + self.lambda1 * quantization_loss.mean()

        return loss


def train(
        test_loader,
        train_loader,
        database_loader,
        code_length,
        args,
):
    """
    Training model.

    Args
        test_loader, database_loader(torch.utils.data.dataloader.DataLoader): Data loader.
        code_length(int): Hashing code length.
        args.device(torch.args.device): GPU or CPU.
        lr(float): Learning rate.
    Returns
        mAP(float): Mean Average Precision.
    """
    # Initialization
    # model = alexnet.load_model(code_length).to(args.device)
    device = args.device
    args.num_train = len(train_loader.dataset)
    args.step_continuation = 20

    model = eval(args.net)(code_length).to(args.device)

    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momen, nesterov=args.nesterov)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.lr_step)
    criterion = DCHLoss(args, code_length)

    losses = AverageMeter()
    start = time.time()
    best_mAP = 0
    for epoch in range(args.max_epoch):
        epoch_start = time.time()

        # criterion.scale = (epoch // args.step_continuation + 1) ** 0.5
        model.train()
        losses.reset()
        for batch, (data, targets, index) in enumerate(train_loader):
            data, targets, index = data.to(args.device), targets.to(args.device), index.to(args.device)
            optimizer.zero_grad()
            u = model(data)
            loss = criterion(u, targets.float(), index)
            losses.update(loss.item())
            loss.backward()
            optimizer.step()
        logger.info('[epoch:{}/{}][loss:{:.6f}]'.format(epoch+1, args.max_epoch, losses.avg))
        scheduler.step()

        if (epoch + 1) % args.val_freq == 0:
            tst_binary, tst_label = compute_result(test_loader, model, device=device)

            # print("calculating dataset binary code.......")\
            db_binary, db_label = compute_result(database_loader, model, device=device)
            # query_code = generate_code(model, test_loader, code_length, args.device)
            # mAP = evaluate.mean_average_precision(
            #     query_code.to(args.device),
            #     B,
            #     test_loader.dataset.get_onehot_targets().to(args.device),
            #     retrieval_targets,
            #     args.device,
            #     args.topk,
            # )

            mAP = CalcTopMap(db_binary.numpy(), tst_binary.numpy(), db_label.numpy(), tst_label.numpy(), args.topk)
            if mAP > best_mAP:
                best_mAP = mAP
                # Save checkpoints
                ret_path = os.path.join('checkpoints', args.info, str(code_length))
                # ret_path = 'checkpoints/' + args.info
                if not os.path.exists(ret_path):
                    os.makedirs(ret_path)
                np.save(os.path.join(ret_path, args.dataset + "-" + str(mAP) + "-db_binary.npy"), db_binary.numpy())
                torch.save(model.state_dict(), os.path.join(ret_path, args.dataset + "-" + str(mAP) + "-model.pt"))
                model = model.to(args.device)
            logger.info('[epoch:{}/{}][code_length:{}][dataset:{}][mAP:{:.5f}][best_mAP:{:.5f}]'.format(epoch+1, 
                args.max_epoch, code_length, args.dataset, mAP, best_mAP))
    logger.info('[Training time:{:.2f}]'.format(time.time()-start))


    return best_mAP
