import torch
import torch.optim as optim
import os
import time
import models.alexnet as alexnet
import utils.evaluate as evaluate
import models.resnet as resnet
import models.exchnet as exchnet

from loguru import logger
from models.adsh_loss import ADSH_Loss
from models.exchnet_loss import Exch_Loss
from data.data_loader import sample_dataloader
from utils import AverageMeter


def train(
        query_dataloader,
        retrieval_dataloader,
        code_length,
        args
        # device,
        # lr,
        # args.max_iter,
        # args.max_epoch,
        # args.num_samples,
        # args.batch_size,
        # args.root,
        # dataset,
        # args.gamma,
        # args.topk,
):
    """
    Training model.

    Args
        query_dataloader, retrieval_dataloader(torch.utils.data.dataloader.DataLoader): Data loader.
        code_length(int): Hashing code length.
        device(torch.device): GPU or CPU.
        lr(float): Learning rate.
        args.max_iter(int): Number of iterations.
        args.max_epoch(int): Number of epochs.
        num_train(int): Number of sampling training data points.
        args.batch_size(int): Batch size.
        args.root(str): Path of dataset.
        dataset(str): Dataset name.
        args.gamma(float): Hyper-parameters.
        args.topk(int): args.Topk k map.

    Returns
        mAP(float): Mean Average Precision.
    """
    # Initialization
    # model = alexnet.load_model(code_length).to(device)
    # model = resnet.resnet50(pretrained=True, num_classes=code_length).to(device)
    num_classes, att_size, feat_size = args.num_classes, 4, 2048
    model = exchnet.exchnet(code_length=code_length, num_classes=num_classes, att_size=att_size, feat_size=feat_size,
                            device=args.device, pretrained=args.pretrain).to(args.device)
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momen, nesterov=args.nesterov)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.lr_step)
    criterion = Exch_Loss(code_length, args.device, lambd_sp=1.0, lambd_ch=1.0)

    criterion.quanting = args.quan_loss
    num_retrieval = len(retrieval_dataloader.dataset)
    U = torch.zeros(args.num_samples, code_length).to(args.device)
    B = torch.randn(num_retrieval, code_length).to(args.device)
    # B = torch.zeros(num_retrieval, code_length).to(args.device)
    retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets().to(args.device)
    C = torch.zeros((num_classes, att_size, feat_size)).to(args.device)
    start = time.time()
    best_mAP = 0
    for it in range(args.max_iter):
        iter_start = time.time()
        # Sample training data for cnn learning
        train_dataloader, sample_index = sample_dataloader(retrieval_dataloader, args.num_samples, args.batch_size, args.root, args.dataset)

        # Create Similarity matrix
        train_targets = train_dataloader.dataset.get_onehot_targets().to(args.device)
        S = (train_targets @ retrieval_targets.t() > 0).float()
        S = torch.where(S == 1, torch.full_like(S, 1), torch.full_like(S, -1))

        # Soft similarity matrix, benefit to converge
        r = S.sum() / (1 - S).sum()
        S = S * (1 + r) - r
        cnn_losses, hash_losses, quan_losses,  sp_losses, ch_losses, align_losses = AverageMeter(), \
            AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

        # Training CNN model
        for epoch in range(args.max_epoch):
            cnn_losses.reset()
            hash_losses.reset()
            quan_losses.reset()
            sp_losses.reset()
            ch_losses.reset()
            align_losses.reset()
            for batch, (data, targets, index) in enumerate(train_dataloader):
                data, targets, index = data.to(args.device), targets.to(args.device), index.to(args.device)
                optimizer.zero_grad()
                F, sp_v, ch_v, avg_local_f = model(data, targets)
                U[index, :] = F.data
                batch_anchor_local_f = C[torch.argmax(targets, dim=1)]
                # print(index)
                cnn_loss, hash_loss, quan_loss,  sp_loss, ch_loss, align_loss = criterion(F, 
                    B, S[index, :], sample_index[index], sp_v, ch_v, avg_local_f, batch_anchor_local_f)
                cnn_losses.update(cnn_loss.item())
                hash_losses.update(hash_loss.item())
                quan_losses.update(quan_loss.item())
                sp_losses.update(sp_loss.item())
                ch_losses.update(ch_loss.item())
                align_losses.update(align_loss.item())
                # print(ch_v)
                cnn_loss.backward()
                optimizer.step()
            logger.info('[epoch:{}/{}][cnn_loss:{:.6f}][h_loss:{:.6f}][q_loss:{:.6f}][s_loss:{:.4f}][c_loss:{:.4f}][a_loss:{:.4f}]'.format(
                epoch+1, args.max_epoch, cnn_losses.avg, hash_losses.avg, quan_losses.avg, sp_losses.avg, ch_losses.avg, align_losses.avg))
        scheduler.step()
        # Update B
        expand_U = torch.zeros(B.shape).to(args.device)
        expand_U[sample_index, :] = U
        if args.quan_loss:
            B = solve_dcc_adsh(B, U, expand_U, S, code_length, args.gamma)
        else:
            B = solve_dcc_exch(B, U, expand_U, S, code_length, args.gamma)

        # Update C
        if (it + 1) >= args.align_step:
            model.exchanging = True
            # criterion.aligning = True
            model.eval()
            with torch.no_grad():
                C = torch.zeros((num_classes, att_size, feat_size)).to(args.device)
                feat_cnt = torch.zeros((num_classes, 1, 1)).to(args.device)
                for batch, (data, targets, index) in enumerate(retrieval_dataloader):
                    data, targets, index = data.to(args.device), targets.to(args.device), index.to(args.device)
                    _, _, _, avg_local_f = model(data, targets)
                    class_idx = targets.argmax(dim=1)
                    for i in range(targets.shape[0]):
                        C[class_idx[i]] += avg_local_f[i]
                        feat_cnt[class_idx[i]] += 1
                C /= feat_cnt
                model.anchor_local_f = C
            model.train()

        # Total loss
        iter_loss = calc_loss(U, B, S, code_length, sample_index, args.gamma)
        # logger.debug('[iter:{}/{}][loss:{:.2f}][iter_time:{:.2f}]'.format(it+1, args.max_iter, iter_loss, time.time()-iter_start))
        logger.info('[iter:{}/{}][loss:{:.6f}][iter_time:{:.2f}]'.format(it+1, args.max_iter, iter_loss, time.time()-iter_start))

        # Evaluate
        if (it + 1) % 1 == 0:
            query_code = generate_code(model, query_dataloader, code_length, args.device)
            mAP = evaluate.mean_average_precision(
                query_code.to(args.device),
                B,
                query_dataloader.dataset.get_onehot_targets().to(args.device),
                retrieval_targets,
                args.device,
                args.topk,
            )
            if mAP > best_mAP:
                best_mAP = mAP
            # Save checkpoints
                ret_path = os.path.join('checkpoints', args.info, str(code_length))
                if not os.path.exists(ret_path):
                    os.makedirs(ret_path)
                torch.save(query_code.cpu(), os.path.join(ret_path, 'query_code.t'))
                torch.save(B.cpu(), os.path.join(ret_path, 'database_code.t'))
                torch.save(query_dataloader.dataset.get_onehot_targets, os.path.join(ret_path, 'query_targets.t'))
                torch.save(retrieval_targets.cpu(), os.path.join(ret_path, 'database_targets.t'))
                torch.save(model.cpu(), os.path.join(ret_path, 'model.t'))
                model = model.to(args.device)
            logger.info('[iter:{}/{}][code_length:{}][mAP:{:.5f}][best_mAP:{:.5f}]'.format(it+1, args.max_iter, code_length, mAP, best_mAP))

    logger.info('[Training time:{:.2f}]'.format(time.time()-start))

    return best_mAP


def solve_dcc_exch(B, U, expand_U, S, code_length, gamma):
    """
    Solve DCC problem.
    """
    Q = (code_length * S).t() @ U

    for bit in range(code_length):
        q = Q[:, bit]
        u = U[:, bit]
        B_prime = torch.cat((B[:, :bit], B[:, bit+1:]), dim=1)
        U_prime = torch.cat((U[:, :bit], U[:, bit+1:]), dim=1)

        # B[:, bit] = (B_prime @ U_prime.t() @ u.t() - q.t()).sign()
        B[:, bit] = (q.t() - B_prime @ U_prime.t() @ u.t()).sign()

    return B


def solve_dcc_adsh(B, U, expand_U, S, code_length, gamma):
    """
    Solve DCC problem.
    """
    Q = (code_length * S).t() @ U + gamma * expand_U

    for bit in range(code_length):
        q = Q[:, bit]
        u = U[:, bit]
        B_prime = torch.cat((B[:, :bit], B[:, bit+1:]), dim=1)
        U_prime = torch.cat((U[:, :bit], U[:, bit+1:]), dim=1)

        B[:, bit] = (q.t() - B_prime @ U_prime.t() @ u.t()).sign()

    return B



def calc_loss(U, B, S, code_length, omega, gamma):
    """
    Calculate loss.
    """
    hash_loss = ((code_length * S - U @ B.t()) ** 2).sum()
    quan_loss = ((U - B[omega, :]) ** 2).sum()
    loss = (hash_loss + gamma * quan_loss) / (U.shape[0] * B.shape[0])

    return loss.item()


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code

    Args
        dataloader(torch.utils.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.

    Returns
        code(torch.Tensor): Hash code.
    """
    model.eval()
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length]).to(device)
        for batch, (data, targets, index) in enumerate(dataloader):
            data, targets, index = data.to(device), targets.to(device), index.to(device)
            hash_code, _, _, _ = model(data, targets)
            code[index, :] = hash_code.sign()
    model.train()
    return code
