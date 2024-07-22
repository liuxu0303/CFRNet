from torch.utils.data import DataLoader
# local modules
from .dataset_dsec import Sequence
from utils.data import concatenate_subfolders, concatenate_datasets



class HDF5DataLoader(DataLoader):
    """
    """
    def __init__(self, data_file, batch_size, shuffle=True, num_workers=1,
                 pin_memory=True, sequence_kwargs={}):
        dataset = concatenate_datasets(data_file, Sequence, sequence_kwargs)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
