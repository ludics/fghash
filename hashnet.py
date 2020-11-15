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

class HashNetLoss(torch.nn.Module):
    def __init__(self, args, bit):
        super(HashNetLoss, self).__init__()
        self.U = torch.zeros(args.num_train, bit).float().to(args.device)
        self.Y = torch.zeros(args.num_train, args.num_classes).float().to(args.device)

        self.scale = 1
        self.alpha = 0.1

    def forward(self, u, y, ind):
        u = torch.tanh(self.scale * u)

        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        similarity = (y @ self.Y.t() > 0).float()
        dot_product = self.alpha * u @ self.U.t()

        mask_positive = similarity.data > 0
        mask_negative = similarity.data <= 0

        exp_loss = (1 + (-dot_product.abs()).exp()).log() + dot_product.clamp(min=0) - similarity * dot_product

        # weight
        S1 = mask_positive.float().sum()
        S0 = mask_negative.float().sum()
        S = S0 + S1
        exp_loss[mask_positive] = exp_loss[mask_positive] * (S / S1)
        exp_loss[mask_negative] = exp_loss[mask_negative] * (S / S0)

        loss = exp_loss.sum() / S

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

    model = AlexNet(code_length).to(args.device)

    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momen, nesterov=args.nesterov)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.lr_step)
    criterion = HashNetLoss(args, code_length)

    losses = AverageMeter()
    start = time.time()
    best_mAP = 0
    for epoch in range(args.max_epoch):
        epoch_start = time.time()

        criterion.scale = (epoch // args.step_continuation + 1) ** 0.5
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
        logger.info('[epoch:{}/{}][scale:{:.3f}][loss:{:.6f}]'.format(epoch+1, args.max_epoch,
                        criterion.scale, losses.avg))
        scheduler.step()

        if (epoch + 1) % 1 == 0:
            tst_binary, tst_label = compute_result(test_loader, model, device=device)

            # print("calculating dataset binary code.......")\
            db_binary, db_label = compute_result(database_loader, model, device=device)
            query_code = generate_code(model, test_loader, code_length, args.device)
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
