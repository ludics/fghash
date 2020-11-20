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

class CSQLoss(torch.nn.Module):
    def __init__(self, args, bit):
        super(CSQLoss, self).__init__()
        self.is_single_label = args.dataset not in {"nuswide_21", "nuswide_21_m", "coco"}
        self.hash_targets = self.get_hash_targets(args.n_class, bit).to(args.device)
        self.multi_label_random_center = torch.randint(2, (bit,)).float().to(args.device)
        self.criterion = torch.nn.BCELoss().to(args.device)

    def forward(self, u, y, ind):
        u = u.tanh()
        hash_center = self.label2center(y)
        center_loss = self.criterion(0.5 * (u + 1), 0.5 * (hash_center + 1))

        Q_loss = (u.abs() - 1).pow(2).mean()
        return center_loss + args.lambd * Q_loss

    def label2center(self, y):
        if self.is_single_label:
            hash_center = self.hash_targets[y.argmax(axis=1)]
        else:
            # to get sign no need to use mean, use sum here
            center_sum = y @ self.hash_targets
            random_center = self.multi_label_random_center.repeat(center_sum.shape[0], 1)
            center_sum[center_sum == 0] = random_center[center_sum == 0]
            hash_center = 2 * (center_sum > 0).float() - 1
        return hash_center

    # use algorithm 1 to generate hash centers
    def get_hash_targets(self, n_class, bit):
        H_K = hadamard(bit)
        H_2K = np.concatenate((H_K, -H_K), 0)
        hash_targets = torch.from_numpy(H_2K[:n_class]).float()

        if H_2K.shape[0] < n_class:
            hash_targets.resize_(n_class, bit)
            for k in range(20):
                for index in range(H_2K.shape[0], n_class):
                    ones = torch.ones(bit)
                    # Bernouli distribution
                    sa = random.sample(list(range(bit)), bit // 2)
                    ones[sa] = -1
                    hash_targets[index] = ones
                # to find average/min  pairwise distance
                c = []
                for i in range(n_class):
                    for j in range(n_class):
                        if i < j:
                            TF = sum(hash_targets[i] != hash_targets[j])
                            c.append(TF)
                c = np.array(c)

                # choose min(c) in the range of K/4 to K/3
                # see in https://github.com/yuanli2333/Hadamard-Matrix-for-hashing/issues/1
                # but it is hard when bit is  small
                if c.min() > bit / 4 and c.mean() >= bit / 2:
                    print(c.min(), c.mean())
                    break
        return hash_targets



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

    model = eval(args.net)(code_length).to(device)

    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momen, nesterov=args.nesterov)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.lr_step)
    
    criterion = CSQLoss(args, code_length)

    losses = AverageMeter()
    start = time.time()
    best_mAP = 0
    for epoch in range(args.max_epoch):
        epoch_start = time.time()

        # criterion.scale = (epoch // args.step_continuation + 1) ** 0.5
        
        model.train()
        losses.reset()
        for batch, (data, targets, index) in enumerate(train_loader):
            data, targets, index = data.to(device), targets.to(device), index.to(device)
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
                model = model.to(device)
            logger.info('[epoch:{}/{}][code_length:{}][dataset:{}][mAP:{:.5f}][best_mAP:{:.5f}]'.format(epoch+1, 
                args.max_epoch, code_length, args.dataset, mAP, best_mAP))
    logger.info('[Training time:{:.2f}]'.format(time.time()-start))


    return best_mAP
