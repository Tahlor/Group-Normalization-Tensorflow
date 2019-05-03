#import trainer
#import time
from tensorboard_aggregator import aggregator

path = r"/media/data/GitHub/Group-Normalization-Tensorflow/train_dir/first_try/baseline"

"""
conda activate batch_norm
tensorboard --logdir /media/data/GitHub/Group-Normalization-Tensorflow/train_dir
"""

if False:
    max_step = 5000
    runs = ["--dataset CIFAR10 --norm_type batch_skew --prefix baseline_skew --max_training_step {} --batch_size 64".format(max_step),
            "--dataset CIFAR10 --norm_type batch_mine --prefix baseline_mine --max_training_step {} --batch_size 64".format(max_step)]
    # try:
    if True:
        for run in runs:
            for i in range(0,5):
                # --checkpoint ./train_dir/main
                trainer.main(run)
                time.sleep(10)


aggregator.main(path, subpaths=[''], output="csv") #, output, subpaths)