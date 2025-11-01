from random import Random
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist


class Partition():
    def __init__(self, data, index):
        self.data = data
        self.index = index
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, index):
        '''Given index, get the data according to the partitioned index'''
        # BEGIN ASSIGN5_1_1
        data_idx = self.index[index]
        return self.data[data_idx]
        # END ASSIGN5_1_1

class DataPartitioner():
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        ''' Create indices for different partitions
        1. Create indices and use `rng` to shuffle indices
        2. Create different partitions of indices according to `sizes` and store in `self.partitions`
        '''
        # BEGIN ASSIGN5_1_1
        # Create a list of indices for the dataset
        data_len = len(data)
        indices = list(range(data_len))
        rng.shuffle(indices)
        
        # Partition the indices according to sizes
        from_idx = 0
        for frac in sizes:
            to_idx = from_idx + int(frac * data_len)
            self.partitions.append(indices[from_idx:to_idx])
            from_idx = to_idx
        # END ASSIGN5_1_1

    def use(self, partition):
        ''' Return a simple dataset class `Partiton` by original data and partitioned indices

        Just one line of code. Think it simply.
        '''
        # BEGIN ASSIGN5_1_1
        return Partition(self.data, self.partitions[partition])
        # END ASSIGN5_1_1

def partition_dataset(rank, world_size, dataset, batch_size=128, collate_fn=None):
    """ Partitioning training dataset of the Machine Translation

    Returns:
        DataLoader: partitioned dataloader
    
    Hint:
    1. Calculate the partitioned batch size
    2. Create a partitioner class `DataPartitioner` with dataset and the list of partitioned sizes
    3. Get the current partition dataset given `rank`, use the `use` function in DataPartitioner
    4. Wrap the dataset with `DataLoader`, remember to customize the `collate_fn`
    """
    # BEGIN ASSIGN5_1
    # 1. Calculate the partitioned batch size for each GPU
    partitioned_batch_size = batch_size // world_size
    
    # 2. Create a list of equal partition sizes for all GPUs
    sizes = [1.0 / world_size for _ in range(world_size)]
    
    # 3. Create DataPartitioner with dataset and partition sizes
    partitioner = DataPartitioner(dataset, sizes=sizes)
    
    # 4. Get the current partition for this rank
    partition = partitioner.use(rank)
    
    # 5. Wrap with DataLoader
    dataloader = DataLoader(partition, batch_size=partitioned_batch_size, collate_fn=collate_fn)
    
    return dataloader
    # END ASSIGN5_1
