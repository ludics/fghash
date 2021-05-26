import torch
import argparse
import adsh
import adsh_exchnet
import adsh_hr

from loguru import logger
from data.data_loader import load_data
import os

import hashnet
import csq
import dch
import cirhash

def run():
    args = load_config()
    log_path = 'logs/' + args.arch + '-' + args.net + '/' + args.info
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logger.add(log_path + '-{time}.log', rotation='500 MB', level='INFO')
    logger.info(args)

    torch.backends.cudnn.benchmark = True

    # Load dataset
    if args.pksampler:
        query_dataloader, train_dataloader, retrieval_dataloader = load_data(
            args.dataset,
            args.root,
            args.num_query,
            args.num_samples,
            args.batch_size,
            args.num_workers,
            'PK'
        )
    else:
        query_dataloader, train_dataloader, retrieval_dataloader = load_data(
            args.dataset,
            args.root,
            args.num_query,
            args.num_samples,
            args.batch_size,
            args.num_workers
        )
    if args.arch == 'baseline':
        net_arch = adsh
    elif args.arch == 'exchnet':
        net_arch = adsh_exchnet
    elif args.arch == 'hrnet':
        net_arch = adsh_hr
    elif args.arch == 'hashnet':
        net_arch = hashnet
    elif args.arch == 'csq':
        net_arch = csq
    elif args.arch == 'dch':
        net_arch = dch
    elif args.arch == 'cirhash':
        net_arch = cirhash
    for code_length in args.code_length:
        mAP = net_arch.train(query_dataloader, train_dataloader, retrieval_dataloader, code_length, args)
            # args.device,
            # args.lr,
            # args.max_iter,
            # args.max_epoch,
            # args.num_samples,
            # args.batch_size,
            # args.root,
            # args.dataset,
            # args.gamma,
            # args.topk,
        logger.info('[code_length:{}][map:{:.4f}]'.format(code_length, mAP))


def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='ADSH_PyTorch')
    parser.add_argument('--dataset',
                        help='Dataset name.')
    parser.add_argument('--root',
                        help='Path of dataset')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='Batch size.(default: 64)')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Learning rate.(default: 1e-4)')
    parser.add_argument('--wd', default=1e-5, type=float,
                        help='Weight Decay.(default: 1e-5)')
    parser.add_argument('--optim', default='Adam', type=str,
                        help='Optimizer')
    parser.add_argument('--code-length', default='12,24,32,48', type=str,
                        help='Binary hash code length.(default: 12,24,32,48)')
    parser.add_argument('--max-iter', default=50, type=int,
                        help='Number of iterations.(default: 50)')
    parser.add_argument('--max-epoch', default=3, type=int,
                        help='Number of epochs.(default: 3)')
    parser.add_argument('--num-query', default=1000, type=int,
                        help='Number of query data points.(default: 1000)')
    parser.add_argument('--num-samples', default=2000, type=int,
                        help='Number of sampling data points.(default: 2000)')
    parser.add_argument('--num-workers', default=0, type=int,
                        help='Number of loading data threads.(default: 0)')
    parser.add_argument('--topk', default=-1, type=int,
                        help='Calculate map of top k.(default: all)')
    parser.add_argument('--gpu', default=None, type=int,
                        help='Using gpu.(default: False)')
    parser.add_argument('--gamma', default=200, type=float,
                        help='Hyper-parameter.(default: 200)')
    parser.add_argument('--info', default='Trivial',
                        help='Train info')
    parser.add_argument('--arch', default='baseline',
                        help='Net arch')
    parser.add_argument('--net', default='AlexNet',
                        help='Net arch')
    parser.add_argument('--save_ckpt', default='checkpoints/',
                        help='result_save')
    parser.add_argument('--lr-step', default='30,45', type=str,
                        help='lr decrease step.(default: 30,45)')
    parser.add_argument('--align-step', default=50, type=int,
                        help='Step of start aligning.(default: 50)')
    parser.add_argument('--pretrain', action='store_true',
                        help='Using image net pretrain')
    parser.add_argument('--quan-loss', action='store_true',
                        help='Using quan_loss')
    parser.add_argument('--lambd-sp', default=0.1, type=float,
                        help='Hyper-parameter.(default: 1)')
    parser.add_argument('--lambd-ch', default=0.1, type=float,
                        help='Hyper-parameter.(default: 1)')
    parser.add_argument('--lambd', default=0.1, type=float,
                        help='Hyper-parameter.(default: 1)')
    parser.add_argument('--momen', default=0.9, type=float,
                        help='Hyper-parameter.(default: 0.9)')
    parser.add_argument('--nesterov', action='store_true',
                        help='Using SGD nesterov')
    parser.add_argument('--cfg', default='experiments/cls_hrnet_w44_sgd_lr5e-2_wd1e-4_bs32_x100.yaml' , type=str,
                        help='HRNet config')
    parser.add_argument('--num-classes', default=200, type=int,
                        help='Number of classes.(default: 200)')
    parser.add_argument('--val-freq', default=10, type=int,
                        help='Number of validate frequency.(default: 10)')
    parser.add_argument('--cauchy-gamma', default=20.0, type=float,
                        help='Hyper-parameter.(default: 1)')
    parser.add_argument('--margin', default=0.1, type=float,
                        help='Hyper-parameter.(default: 1)')
    parser.add_argument('--pksampler', action='store_true',
                        help='Using image net pretrain')
    parser.add_argument('--pk', default=80, type=int,
                        help='Number of epochs.(default: 3)')
    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)

    # Hash code length
    args.code_length = list(map(int, args.code_length.split(',')))
    args.lr_step = list(map(int, args.lr_step.split(',')))

    return args


if __name__ == '__main__':
    run()
