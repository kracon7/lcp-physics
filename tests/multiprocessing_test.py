import torch
from torch import multiprocessing as mp
from torch.multiprocessing import Pool
from torchvision.models import resnet50
torch.set_num_threads(1)

class C():
    """docstring for C"""
    def __init__(self):
        self.x = torch.tensor(10).float()


def main():
    args = []
    for i in range(10):
        args.append([C(), i])

    with Pool(5) as p:
        worlds_after = p.starmap(run_worker, args)

    for i in range(len(args)):
        print(args[i][0].x)


def run_worker(c, i):
    c.x += i

if __name__ == "__main__":
    main()