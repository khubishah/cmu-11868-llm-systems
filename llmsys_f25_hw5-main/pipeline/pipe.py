from typing import Any, Iterable, Iterator, List, Optional, Union, Sequence, Tuple, cast

import torch
from torch import Tensor, nn
import torch.autograd
import torch.cuda
from .worker import Task, create_workers
from .partition import _split_module

def _clock_cycles(num_batches: int, num_partitions: int) -> Iterable[List[Tuple[int, int]]]:
    '''Generate schedules for each clock cycle.

    An example of the generated schedule for m=3 and n=3 is as follows:
    
    k (i,j) (i,j) (i,j)
    - ----- ----- -----
    0 (0,0)
    1 (1,0) (0,1)
    2 (2,0) (1,1) (0,2)
    3       (2,1) (1,2)
    4             (2,2)

    where k is the clock number, i is the index of micro-batch, and j is the index of partition.

    Each schedule is a list of tuples. Each tuple contains the index of micro-batch and the index of partition.
    This function should yield schedules for each clock cycle.
    '''
    # BEGIN ASSIGN5_2_1
    # Pipeline parallelism: at clock k, process all (i,j) where i+j=k
    # Total clock cycles needed: num_batches + num_partitions - 1
    for clock in range(num_batches + num_partitions - 1):
        schedule = []
        # For each clock cycle, find all valid (batch_idx, partition_idx) pairs
        # where batch_idx + partition_idx == clock
        for partition_idx in range(num_partitions):
            batch_idx = clock - partition_idx
            # Check if this is a valid batch index
            if 0 <= batch_idx < num_batches:
                schedule.append((batch_idx, partition_idx))
        yield schedule
    # END ASSIGN5_2_1

class Pipe(nn.Module):
    def __init__(
        self,
        module: nn.ModuleList,
        split_size: int = 1,
    ) -> None:
        super().__init__()

        self.split_size = int(split_size)
        self.partitions, self.devices = _split_module(module)
        (self.in_queues, self.out_queues) = create_workers(self.devices)

    def forward(self, x):
        ''' Forward the input x through the pipeline. The return value should be put in the last device.

        Hint:
        1. Divide the input mini-batch into micro-batches.
        2. Generate the clock schedule.
        3. Call self.compute to compute the micro-batches in parallel.
        4. Concatenate the micro-batches to form the mini-batch and return it.
        
        Please note that you should put the result on the last device. Putting the result on the same device as input x will lead to pipeline parallel training failing.
        '''
        # BEGIN ASSIGN5_2_2
        # 1. Split input into microbatches
        microbatches = list(x.split(self.split_size, dim=0))
        num_batches = len(microbatches)
        num_partitions = len(self.partitions)
        
        # 2. Generate clock schedule
        schedules = _clock_cycles(num_batches, num_partitions)
        
        # 3. Process microbatches through the pipeline
        for schedule in schedules:
            self.compute(microbatches, schedule)
        
        # 4. Concatenate results and put on last device
        result = torch.cat(microbatches, dim=0)
        last_device = self.devices[-1]
        result = result.to(last_device)
        
        return result
        # END ASSIGN5_2_2

    def compute(self, batches, schedule: List[Tuple[int, int]]) -> None:
        '''Compute the micro-batches in parallel.

        Hint:
        1. Retrieve the partition and microbatch from the schedule.
        2. Use Task to send the computation to a worker. 
        3. Use the in_queues and out_queues to send and receive tasks.
        4. Store the result back to the batches.
        '''
        partitions = self.partitions
        devices = self.devices

        # BEGIN ASSIGN5_2_2
        # Submit all tasks in the schedule to their respective workers
        for batch_idx, partition_idx in schedule:
            # Get the microbatch and partition
            microbatch = batches[batch_idx]
            partition = partitions[partition_idx]
            device = devices[partition_idx]
            
            # Move microbatch to the correct device and create compute function
            def compute_fn(mb=microbatch, part=partition, dev=device):
                mb = mb.to(dev)
                return part(mb)
            
            # Create task and submit to worker
            task = Task(compute_fn)
            self.in_queues[partition_idx].put(task)
        
        # Retrieve results from workers
        for batch_idx, partition_idx in schedule:
            success, data = self.out_queues[partition_idx].get()
            
            if not success:
                # Error occurred, re-raise the exception
                exc_info = data
                raise exc_info[1].with_traceback(exc_info[2])
            
            # Extract result and store back in batches
            task, result = data
            batches[batch_idx] = result
        # END ASSIGN5_2_2

