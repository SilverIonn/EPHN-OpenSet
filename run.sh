CUDA_VISIBLE_DEVICES='0' screen  python3 run.py --Data LMK --model R18 --dim 64 --lr 1e-2 --method EPSHN --imgsize 128 --c 16 --g 8 --n 128 --ep 20 --w 0.1
