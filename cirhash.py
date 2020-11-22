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


from typing import Tuple

from torch import nn, Tensor


def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    label = torch.argmax(label, axis=1)
    # print(label)
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss


class CirHashLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(CirHashLoss, self).__init__()
        self.cauchy_gamma = config.cauchy_gamma
        self.lambda1 = config.lambd
        self.K = bit
        self.one = torch.ones((config.batch_size, bit)).to(config.device)
        self.circleloss = CircleLoss(m=config.margin, gamma=config.gamma)

    def d(self, hi, hj):
        inner_product = hi @ hj.t()
        norm = hi.pow(2).sum(dim=1, keepdim=True).pow(0.5) @ hj.pow(2).sum(dim=1, keepdim=True).pow(0.5).t()
        cos = inner_product / norm.clamp(min=0.0001)
        # formula 6
        return (1 - cos.clamp(max=0.99)) * self.K / 2

    def forward(self, u, y, ind):
        norm_u = nn.functional.normalize(u)
        cir_loss = self.circleloss(*convert_label_to_similarity(norm_u, y))
        # s = (y @ y.t() > 0).float()

        # if (1 - s).sum() != 0 and s.sum() != 0:
        #     # formula 2
        #     positive_w = s * s.numel() / s.sum()
        #     negative_w = (1 - s) * s.numel() / (1 - s).sum()
        #     w = positive_w + negative_w
        # else:
        #     # maybe |S1|==0 or |S2|==0
        #     w = 1

        # d_hi_hj = self.d(u, u)
        # # formula 8
        # cauchy_loss = w * (s * torch.log(d_hi_hj / self.gamma) + torch.log(1 + self.gamma / d_hi_hj))
        # formula 9
        quant_loss = torch.log(1 + self.d(u.abs(), self.one) / self.cauchy_gamma)
        # formula 7
        loss = cir_loss.mean() + self.lambda1 * quant_loss.mean()

        return loss, cir_loss, quant_loss


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
    criterion = CirHashLoss(args, code_length)

    losses = AverageMeter()
    cir_losses = AverageMeter()
    quant_losses = AverageMeter()
    start = time.time()
    best_mAP = 0
    for epoch in range(args.max_epoch):
        epoch_start = time.time()

        # criterion.scale = (epoch // args.step_continuation + 1) ** 0.5
        model.train()
        losses.reset()
        cir_losses.reset()
        quant_losses.reset()
        for batch, (data, targets, index) in enumerate(train_loader):
            data, targets, index = data.to(args.device), targets.to(args.device), index.to(args.device)
            optimizer.zero_grad()
            u = model(data)
            loss, cir_loss, quant_loss = criterion(u, targets.float(), index)
            losses.update(loss.item())
            cir_losses.update(cir_loss.item())
            quant_losses.update(quant_loss.item())
            loss.backward()
            optimizer.step()
        logger.info('[epoch:{}/{}][loss:{:.6f}][cir_loss:{:.6f}][quant_loss:{:.6f}]'.format(epoch+1, args.max_epoch, losses.avg, cir_losses.avg, args.lambd * quant_losses.avg))
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
