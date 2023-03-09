python train.py --module part_ae \
                --category easyscene \
                --resolution 16 \
                --nr_epochs 80 \
                --batch_size 80 \
                --lr 5e-4 \
                --lr_step_size 350 \
                --save_frequency 50 \
                -g 0,3 \
                --vis
python train.py --module part_ae \
                --category easyscene \
                --resolution 32 \
                --nr_epochs 150 \
                --batch_size 40 \
                --lr 5e-4 \
                --lr_step_size 350 \
                --save_frequency 50 \
                -g 0,3 \
                --vis \
                --continue
python train.py --module part_ae \
                --category easyscene \
                --resolution 64 \
                --nr_epochs 500 \
                --batch_size 25 \
                --lr 5e-4 \
                --lr_step_size 350 \
                --save_frequency 50 \
                -g 0,3 \
                --vis \
                --continue
