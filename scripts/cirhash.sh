python run.py --dataset cub-2011 --root dataset/CUB_200_2011 --info cirhash-cub-test --arch cirhash \
--net ResNet --batch-size 128 --max-epoch 750 --val-freq 25 --code-length 24 --num-classes 200 --pretrain  \
--gpu 2 --lr 1e-4 --wd 1e-5 --optim Adam --lr-step 250,500 \
--cauchy-gamma 20 --lambd 5 --gamma 80 --margin 0.4 --pksampler
