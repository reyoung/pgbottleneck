import torch.distributed as dist
import sys
import torch
import itertools
from typing import List
import time
import math

def order_generator(world_size: int):
    visited = set()
    for order in itertools.permutations(range(world_size)):
        slices = [order[i*2]*world_size + order[i*2+1]   for i in range(len(order)//2)]
        if any((item in visited for item in slices)):
            continue
        for item in slices:
            visited.add(item)
        
        yield order

def time_to_speed(t: torch.Tensor, size:int, n_iter:int):
    total_size = size * 4 * n_iter
    return (total_size / t).to("cpu")



def main():
    """
    Usage:

    torchrun --nnodes ${WORLD_SIZE}  --master_addr ${MASTER_ADDR} \
        --master_port ${MASTER_PORT} --nproc_per_node 1 --node_rank ${NODE_RANK}\
        main.py gloo
    
    Or 

    torchrun --nnodes ${WORLD_SIZE}  --master_addr ${MASTER_ADDR} \
        --master_port ${MASTER_PORT} --nproc_per_node 1 --node_rank ${NODE_RANK}\
        main.py nccl

    NOTE: --nproc_per_node should be one

    It will save a `result.pt` which is a 3-D tensor, with shape [WorldSize, 2, WorldSize], which 
    records P2P read/write speed.
    """
    dist.init_process_group(sys.argv[1])

    if sys.argv[1] == "gloo":
        device="cpu"
    else:
        device="cuda:0"

    torch.set_grad_enabled(False)
    world_size: int = dist.get_world_size()
    rank: int = dist.get_rank()
    size = 64*1024
    send = torch.rand(size=(size, ), dtype=torch.float, device=device)
    recv = torch.empty_like(send, device=device)

    write_time = [0 for _ in range(world_size)]
    read_time = [0 for _ in range(world_size)]

    n_iter = 100

    def do_operation(order:List[int]):
        idx = order.index(rank)
        if idx % 2 == 0:
            try:
                dest = order[idx+1]
            except IndexError:
                return
            begin = time.time()
            for _ in range(n_iter):
                dist.send(send, dest)
            write_time[dest] = time.time() - begin
        else:
            src = order[idx-1]
            begin = time.time()
            for _ in range(n_iter):
                dist.recv(recv, src)
            read_time[src] = time.time() - begin

    for order in order_generator(world_size):
        order = list(order)
        do_operation(order)
        dist.barrier()


    time_message = torch.tensor([write_time, read_time], dtype=torch.float, device=device)
    result = [torch.empty_like(time_message, device=device) for _ in range(world_size)]
    dist.all_gather(result, time_message)

    if rank == 0:
        k = math.ceil(world_size*0.1)
        to_save = []
        for r, msg in enumerate(result):
            msg: torch.Tensor = time_to_speed(msg, size, n_iter)
            slowest_write_rank = msg[0].topk(k=k,  largest=False).indices.tolist()
            slowest_read_rank = msg[1].topk(k=k, largest=False).indices.tolist()

            print(f"Rank {r}, slowest_write_to {slowest_write_rank}, slowest_read_from {slowest_read_rank}")
        
            to_save.append(msg)
        
        to_save = torch.stack(to_save)
        torch.save(to_save, "result.pt")



if __name__ == '__main__':
    main()