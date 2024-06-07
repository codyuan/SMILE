from torch.utils import data

class DemoDataset(data.Dataset):
    def __init__(self):
        self.demo_list = []

    def __getitem__(self, index):
        return self.demo_list[index]

    def __len__(self):
        return len(self.demo_list)

    def add(self, demo):
        self.demo_list.append(demo)