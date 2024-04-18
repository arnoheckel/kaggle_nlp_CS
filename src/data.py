from torch.utils.data import Dataset

class NewsDataset(Dataset):
    """Custom-built News dataset"""

    def __init__(self, X, y, device='cpu'):
        """
        Args:
            X, y as Torch tensors
        """
        self.X_train = X
        self.y_train = y
        self.device = device
        

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx].to(self.device), self.y_train[idx].to(self.device)