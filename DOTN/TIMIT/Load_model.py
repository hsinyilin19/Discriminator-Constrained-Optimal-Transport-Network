import os
import torch
import torch.nn as nn
from util import get_filepaths
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
#         nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.3))
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
        
def load_checkoutpoint(D, G, optimizer_S, optimizer_D, optimizer_G, optimizer_G_OT, model_path):
    if os.path.isfile(model_path):
        D.eval()
        G.eval()
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        D.load_state_dict(checkpoint['discriminator'])
        G.load_state_dict(checkpoint['generator'])

        optimizer_S.load_state_dict(checkpoint['optimizer_S'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_G_OT.load_state_dict(checkpoint['optimizer_G_OT'])

        for state in optimizer_S.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        for state in optimizer_D.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        for state in optimizer_G.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        for state in optimizer_G_OT.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        print(f"=> loaded checkpoint '{model_path}' (epoch {epoch})")
        
        return D, G, epoch, best_loss, optimizer_S, optimizer_D, optimizer_G, optimizer_G_OT
    else:
        raise NameError(f"=> no checkpoint found at '{model_path}'")


def Load_model(args, D, G, checkpoint_path):
    criterion = {'mse': nn.MSELoss(), 'l1': nn.L1Loss(), 'l1smooth': nn.SmoothL1Loss()}
    device = torch.device(f'cuda:{args.gpu}')
    criterion = criterion[args.loss_fn].to(device)

    # to GPU
    D, G = D.to(device), G.to(device)

    # Parallelize model to multiple GPUs
    print("Using", torch.cuda.device_count(), "GPU(s)!")
    if torch.cuda.device_count() > 1:
        D = nn.DataParallel(D)
        G = nn.DataParallel(G)

    optimizer_S = torch.optim.Adam(G.parameters(), lr=args.lr_S)
    optimizer_G = torch.optim.Adam(G.parameters(), lr=args.lr_G)
    optimizer_G_OT = torch.optim.Adam(G.parameters(), lr=args.lr_G_OT)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=args.lr_D)

    if args.resume:
        # resume from the last checkpoint autosaved
        D, G, epoch, best_loss, optimizer_S, optimizer_D, optimizer_G, optimizer_G_OT = load_checkoutpoint(D, G,
                                                                                                           optimizer_S,
                                                                                                           optimizer_D,
                                                                                                           optimizer_G,
                                                                                                           optimizer_G_OT,
                                                                                                           checkpoint_path)
    elif args.retrain:
        # retrain from any model assigned
        D, G, epoch, best_loss, optimizer_S, optimizer_D, optimizer_G, optimizer_G_OT = load_checkoutpoint(D, G,
                                                                                                           optimizer_S,
                                                                                                           optimizer_D,
                                                                                                           optimizer_G,
                                                                                                           optimizer_G_OT,
                                                                                                           args.model_path)
    elif args.pretrained_model:
        G.eval()
        print("=> loading checkpoint '{}'".format(args.pretrained_model))
        checkpoint = torch.load(args.pretrained_model)
        epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        G.load_state_dict(checkpoint['generator'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])

        for state in optimizer_G.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        epoch = 0
        best_loss = 100000000000
        D.apply(weights_init)

    else:
        epoch = 0
        best_loss = 100000000000
        D.apply(weights_init)
        G.apply(weights_init)
        
    para_D = count_parameters(D)
    para_G = count_parameters(G)
    print(f'Model parameters- D: {para_D}, G:{para_G}, G/D:{para_G / para_D:.2f}')

    return D, G, epoch, best_loss,  optimizer_S, optimizer_D, optimizer_G, optimizer_G_OT, criterion, device


def Load_data(args):
    # split data
    train_noisy_paths, val_noisy_paths = train_test_split(get_filepaths(args.train_noisy_path, ftype='.pt'), test_size=0.1, random_state=999)
    train_target_noisy_paths, val_target_noisy_paths = train_test_split(get_filepaths(args.target_noisy_path, ftype='.pt'), test_size=0.1, random_state=1000)

    # create train/validate dataset
    train_dataset = CustomDataset(clean_path=args.train_clean_path, train_noisy=train_noisy_paths, target_noisy=train_target_noisy_paths)
    val_dataset = CustomDataset(clean_path=args.target_clean_path, train_noisy=val_target_noisy_paths)

    loaders = {'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True),
               'val': DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)}
    return loaders



class CustomDataset(Dataset):
    def __init__(self, clean_path, train_noisy, target_noisy=None, amplify=1):
        self.train_noisy = train_noisy
        self.train_clean = [os.path.join(clean_path, f.split('/')[-2], f.split('/')[-1]) for f in train_noisy]  # find clean correspondance
        self.target_noisy = target_noisy
        self.amplify = amplify

    def __len__(self):  # return count of sample we have
        return len(self.train_noisy)

    def __getitem__(self, ind):
        y = self.amplify * torch.load(self.train_clean[ind])
        X = self.amplify * torch.load(self.train_noisy[ind])

        # if to provide noisy targets
        if self.target_noisy:
            target_ind = ind % len(self.target_noisy)
            X_t = self.amplify * torch.load(self.target_noisy[target_ind])
            return X, y, X_t

        return X, y
