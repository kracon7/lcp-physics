import torch
from torch.multiprocessing import Pool
torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_num_threads(1)

def f(x):
	return 2 * (x * x).norm()


def main():
    args = []
    for i in range(5):
        args.append(torch.tensor([i], dtype=torch.float64, requires_grad=True))

    with Pool(5) as p:
        y = p.starmap(f, args)


if __name__ == '__main__':
	main()