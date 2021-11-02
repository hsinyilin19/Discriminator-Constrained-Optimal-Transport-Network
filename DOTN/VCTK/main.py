import os, argparse, random, sys
import torch
from Trainer import Trainer
from Load_model import Load_model, Load_data
from util import check_folder
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
from modules import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# fix random
SEED = 999
random.seed(SEED)
torch.manual_seed(SEED)
cudnn.deterministic = True

# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

# data location
data_path = '/work/VCTK_DEMAND_3T_to_SCAFE/'
pt_data_path = '/work/VCTK_DEMAND_3T_to_SCAFE_pt/'

def get_args():
    parser = argparse.ArgumentParser()

    # training data
    parser.add_argument('--train_clean_path', type=str, default=os.path.join(pt_data_path, 'train/clean'))
    parser.add_argument('--train_noisy_path', type=str, default=os.path.join(pt_data_path, 'train/noisy'))

    # target data
    parser.add_argument('--target_clean_path', type=str, default=os.path.join(pt_data_path, 'test/clean'))
    parser.add_argument('--target_noisy_path', type=str, default=os.path.join(pt_data_path, 'test'))

    # for final WAV scores: PESQ, STOI
    parser.add_argument('--WAV_test_clean_path', type=str, default=os.path.join(data_path, 'test/clean'))
    parser.add_argument('--WAV_test_noisy_path', type=str, default=os.path.join(data_path, 'test/noisy'))
    parser.add_argument('--WAV_enhanced_path', type=str, default='./enhanced')    # save enhanced audio

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=1800)
    parser.add_argument('--lr_S', type=float, default=1e-4)
    parser.add_argument('--lr_G', type=float, default=1e-5)
    parser.add_argument('--lr_G_OT', type=float, default=1e-5)
    parser.add_argument('--lr_D', type=float, default=1e-3)
    parser.add_argument('--loss_fn', type=str, default='mse')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--model', type=str, default='DOTN')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--task', type=str, default='SCAFE')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--re_epochs', type=int, default=300)
    parser.add_argument('--period_S', type=str, default=2)  # source domain memory training
    parser.add_argument('--period_D', type=str, default=5)
    parser.add_argument('--period_G', type=str, default=11)
    parser.add_argument('--amplify', type=str, default=1)
    parser.add_argument('--pretrained_model', type=str, default=None)       # provide pretrained SE model G if exists
    parser.add_argument('--model_path', type=str, default=None)             # provide an existing model for testing
    parser.add_argument('--save_path', type=str, default='./saved_model/')  # saved model path
    args = parser.parse_args()
    return args

def get_path(args):
    checkpoint_path = os.path.join(args.save_path, f'{args.model}_epochs{args.epochs}' \
                    f'_{args.optim}_{args.loss_fn}_batch{args.batch_size}_'\
                    f'lr{args.lr_G}_periodS{args.period_S}_periodD{args.period_D}_periodG{args.period_G}.pth.tar')

    score_path = f'./results/{args.model}_epochs{args.epochs}' \
                    f'_{args.optim}_{args.loss_fn}_batch{args.batch_size}_'\
                    f'lr{args.lr_G}_periodS{args.period_S}_periodD{args.period_D}_periodG{args.period_G}.csv'
    log_path = f'./log'
    
    return checkpoint_path, score_path, log_path

if __name__ == '__main__':
    # get current path
    cwd = os.path.dirname(os.path.abspath(__file__))
    # print(cwd)

    # get parameter
    args = get_args()

    # declare path
    checkpoint_path, score_path, log_path = get_path(args)
    
    # tensorboard
    writer = SummaryWriter(log_path)

    # Create model
    discriminator = Discriminator().to(device)
    generator = Generator().to(device)

    discriminator, generator, epoch, best_loss, optimizer_S, optimizer_D, optimizer_G, optimizer_G_OT, criterion, device = Load_model(args, discriminator, generator, checkpoint_path)

    # load data
    loaders = Load_data(args)

    if args.retrain:
        args.epochs = args.re_epochs 
        checkpoint_path, score_path, log_path = get_path(args)
        
    trainer = Trainer(discriminator, generator, args.epochs, epoch, best_loss, optimizer_S, optimizer_D, optimizer_G, optimizer_G_OT,
                      criterion, device, loaders, writer, score_path, args)
    try:
        if args.mode == 'train':
            trainer.train()
        trainer.test()
        
    except KeyboardInterrupt:
        state_dict = {
            'epoch': epoch,
            'discriminator': discriminator.state_dict(),
            'generator': generator.state_dict(),
            'optimizer_S': optimizer_S.state_dict(),
            'optimizer_D': optimizer_D.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_G_OT': optimizer_G_OT.state_dict(),
            'best_loss': best_loss
            }
        check_folder(checkpoint_path)
        torch.save(state_dict, checkpoint_path)
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
