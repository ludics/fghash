python run.py --dataset cub-2011 --root dataset/CUB_200_2011 --info cirhash-cub-test --arch cirhash \
--net ResNet --batch-size 64 --max-epoch 150 --val-freq 5 --code-length 24 --num-classes 200 --pretrain  \
--gpu 2 --lr 1e-4 --wd 1e-5 --optim Adam --lr-step 50,100 \
--cauchy-gamma 20 --lambd 0.1 --gamma 80 --margin 0.4 --pksampler
