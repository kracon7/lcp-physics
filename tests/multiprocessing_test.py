import torch
from torch import multiprocessing as mp
from torchvision.models import resnet50
torch.set_num_threads(1)

def main():
    model = resnet50()

    copy1 = [p.clone() for p in model.parameters()]
    copy2 = [p.clone() for p in model.parameters()]

    processes = []
    for rank in range(2):
        process = mp.Process(target=run_worker, args=(rank,))
        process.start()
        processes.append(process)

    for p in processes:
        p.join()


def run_worker(rank):
    print(f'Started Worker {rank}')
    model = resnet50()
    print('Does get here')
    local_copy = [p.clone() for p in model.parameters()]
    print('And should also get here')


if __name__ == "__main__":
    main()