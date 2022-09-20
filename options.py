import os
import argparse

file_dir = os.path.dirname(__file__)


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Options")

        self.parser.add_argument("--dataset", 
                                 type=int,
                                 help="which dataset will be use, [h36m, coco, mpii, 3dhp] in binary presentation, "
                                      "e.x. 4 = [0, 1, 0, 0]",
                                 default=4)

        self.parser.add_argument("--data_path", 
                                 type=str, 
                                 help="path to the data",
                                 default="./datasets/")

        self.parser.add_argument("--weights_path",
                                 type=str,
                                 help="path to the weights of network",
                                 default="./model/")

        self.parser.add_argument("--subfolder_path",
                                 type=str,
                                 help="path to the subfolder of weights of network",
                                 default="")

        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=32)

        self.parser.add_argument("--pose_lr_rate",
                                 type=float,
                                 help="pose network learning rate",
                                 default=0.0001)

        self.parser.add_argument("--num_key_points",
                                 type=int,
                                 help="number of key points",
                                 default=17)

        self.parser.add_argument("--train_epoch",
                                 type=int,
                                 help="training epoch",
                                 default=2)

        self.parser.add_argument("--diffusion_speed",
                                 type=float,
                                 help="diffusion speed",
                                 default=0.00001)

        self.parser.add_argument("--prediffusion",
                                 type=int,
                                 help="number of steps of pre-diffusion",
                                 default=1000)

        self.parser.add_argument("--task",
                                 type=str,
                                 help="which task to do",
                                 choices=["all", "generation", "train", "inference", "make-distribution", "g", "t", "i", "md"],
                                 default="all")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
