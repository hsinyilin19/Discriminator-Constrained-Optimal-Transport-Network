import os
import numpy as np
import ot
import torch
import librosa
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.io.wavfile import write as audiowrite
from tqdm import tqdm
from util import get_filepaths, check_folder, make_spectrum, recons_spec_phase, cal_score

maxv = np.iinfo(np.int16).max


class Trainer:
    def __init__(self, discriminator, generator, epochs, epoch, best_loss, optimizer_S, optimizer_D, optimizer_G,
                 optimizer_G_OT, criterion, device, loader, writer, score_path, args):

        self.epoch = epoch
        self.epochs = epochs
        self.best_loss = best_loss

        self.D = discriminator.to(device)
        self.G = generator.to(device)

        self.optimizer_S = optimizer_S
        self.optimizer_D = optimizer_D
        self.optimizer_G = optimizer_G
        self.optimizer_G_OT = optimizer_G_OT

        self.device = device
        self.loader = loader
        self.criterion = criterion

        self.epoch_S_loss = 0  # source domain denoise loss
        self.epoch_D_loss = 0
        self.epoch_G_loss = 0
        self.epoch_OT_loss = 0

        self.val_loss = 0

        self.writer = writer
        self.score_path = score_path
        self.args = args

        # training periods
        self.period_S = self.args.period_S
        self.period_D = self.args.period_D
        self.period_G = self.args.period_G

    def save_checkpoint(self):
        state_dict = {
            'epoch': self.epoch,
            'discriminator': self.D.state_dict(),
            'generator': self.G.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_G_OT': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            'best_loss': self.best_loss
        }

        self.save_model = f'{self.args.model}_epoch{self.epoch}_{self.args.optim}_{self.args.loss_fn}_batch{self.args.batch_size}_lr{self.args.lr_G}_periodS{self.args.period_S}_periodD{self.args.period_D}_periodG{self.args.period_G}_valoss{self.val_loss:.3f}.pth.tar'

        # self.save_path = '_'.join(self.model_path.split('_')[:-1]) + '_valoss{}'.format(
        #     self.val_loss) + '_epoch{}'.format(self.epoch) + self.model_path.split('_')[-1]

        check_folder(self.args.save_path)
        torch.save(state_dict, os.path.join(self.args.save_path, self.save_model))

    def OT_loss(self, X_s, X_t, y_s, y_t_pred):
        N = X_s.shape[0]

        C0 = torch.cdist(X_s.reshape((N, -1)), X_t.reshape((N, -1)), p=2).cpu()
        C1 = torch.cdist(y_s.reshape((N, -1)), y_t_pred.reshape((N, -1)), p=2).cpu()

        alpha = 1  # OT source weight in loss
        beta = 1   # OT target weight in loss
        C = alpha * C0 + beta * C1

        γ = ot.emd(ot.unif(N), ot.unif(N), C.detach().numpy())
        γ = torch.from_numpy(γ).float()

        loss = torch.sum(γ * C)

        return loss

    def _val_epoch(self):
        self.val_loss = 0
        self.G.eval()

        with tqdm(total=len(self.loader['val']), desc=f'Validate epoch: {self.epoch}/{self.epochs - 1}', unit='step') as t:
            for X, y in self.loader['val']:
                X, y = X.to(self.device), y.to(self.device)
                y_pred = self.G(X) / self.args.amplify
                loss = F.mse_loss(y_pred.reshape((y.shape[0], y.shape[1], y.shape[2])), y / self.args.amplify, reduction='sum')
                self.val_loss += loss.item()
                t.update(1)

            self.val_loss /= (self.args.batch_size * len(self.loader['val']))  # average batch loss
            t.set_postfix({'Validate loss': '{:.3f}'.format(self.val_loss)})  # display loss

            if (self.epoch > 1) and (self.best_loss > self.val_loss):
                self.save_checkpoint()
                self.best_loss = self.val_loss
                t.set_description(f'model saved at {self.save_model}', refresh=True)

    def train(self):
        while self.epoch < self.epochs:
            # train model
            self.epoch_OT_loss = 0
            self.epoch_S_loss = 0
            self.epoch_D_loss = 0
            self.epoch_G_loss = 0

            self.D.train()
            self.G.train()
            OT_scheduler = ReduceLROnPlateau(self.optimizer_G_OT, mode='min', factor=0.1, patience=4)

            with tqdm(total=len(self.loader['train']), desc=f'Train epoch: {self.epoch}/{self.epochs - 1}',
                      unit='step') as t:
                for batch_idx, (X_s, y_s, X_t) in enumerate(self.loader['train']):
                    X_s, y_s, X_t = X_s.to(self.device), y_s.to(self.device), X_t.to(self.device)

                    batch_D_loss = 'N/A'
                    batch_G_loss = 'N/A'

                    if (batch_idx % self.period_D == 0):

                        ''' ideally for discriminator: D(real) > 0,  D(fake) < 0'''
                        self.optimizer_D.zero_grad()
                        loss_d = -torch.mean(self.D(y_s)) + torch.mean(self.D(self.G(X_t).detach()))
                        loss_d.backward()
                        self.optimizer_D.step()

                        for p in self.D.parameters():
                            p.data.clamp_(-0.001, 0.001)

                        self.epoch_D_loss += loss_d.item()
                        batch_D_loss = str(np.around(loss_d.item(), decimals=3))  # for display

                    if (batch_idx % self.period_G == 0):
                        self.optimizer_G.zero_grad()
                        loss_g = -torch.mean(self.D(self.G(X_t)))
                        loss_g.backward()
                        self.optimizer_G.step()

                        self.epoch_G_loss += loss_g.item()
                        batch_G_loss = str(np.around(loss_g.item(), decimals=3))  # for display

                    if (batch_idx % self.period_S == 0):
                        # loss_s = F.mse_loss(self.G(X_s) / self.args.amplify, y_s / self.args.amplify, reduction='mean')
                        loss_s = self.criterion(self.G(X_s) / self.args.amplify, y_s / self.args.amplify)
                        self.optimizer_S.zero_grad()
                        loss_s.backward()
                        self.optimizer_S.step()
                        self.epoch_S_loss += loss_s.item()

                    ot_loss = self.OT_loss(X_s, X_t, y_s, self.G(X_t)).to(self.device)
                    self.optimizer_G_OT.zero_grad()
                    ot_loss.backward()
                    self.optimizer_G_OT.step()
                    self.epoch_OT_loss += ot_loss.item()

                    # Display losses
                    t.set_postfix({'Batch-D': '{}'.format(batch_D_loss),
                                   'Epoch-D': '{:.3f}'.format(self.epoch_D_loss),
                                   'Batch-G': '{}'.format(batch_G_loss),
                                   'Epoch-G': '{:.3f}'.format(self.epoch_G_loss),
                                   'Epoch-Source': '{:.3f}'.format(self.epoch_S_loss),
                                   'Epoch-OT-Loss': '{:.3f}'.format(self.epoch_OT_loss)}, refresh=True)
                    t.update(1)

            # record train score
            self.writer.add_scalars(
                f'Epoch_D-loss_{self.args.task}_batch{self.args.batch_size}_periodS{self.period_S}_periodD{self.period_D}_periodG{self.period_G}',
                {'train': self.epoch_D_loss}, self.epoch)
            self.writer.add_scalars(
                f'Epoch_G-Loss_{self.args.task}_batch{self.args.batch_size}_periodS{self.period_S}_periodD{self.period_D}_periodG{self.period_G}',
                {'train': self.epoch_G_loss}, self.epoch)
            self.writer.add_scalars(
                f'Epoch-S-Loss_{self.args.task}_batch{self.args.batch_size}_periodS{self.period_S}_periodD{self.period_D}_periodG{self.period_G}',
                {'train': self.epoch_S_loss}, self.epoch)
            self.writer.add_scalars(
                f'Epoch_OT-Loss_{self.args.task}_batch{self.args.batch_size}_periodS{self.period_S}_periodD{self.period_D}_periodG{self.period_G}',
                {'train': self.epoch_OT_loss}, self.epoch)

            # validate model
            self._val_epoch()

            # update scheduler
            OT_scheduler.step(self.val_loss)

            # record validate scores
            self.writer.add_scalars(
                f'val_loss_{self.args.task}_batch{self.args.batch_size}_periodS{self.period_S}_periodD{self.period_D}_periodG{self.period_G}',
                {'train': self.val_loss}, self.epoch)
            self.epoch += 1

    def test(self):
        # # Parallelize model to multiple GPUs
        # print("Testing using", torch.cuda.device_count(), "GPU(s)!")
        # if torch.cuda.device_count() > 1:
        #     self.discriminator = nn.DataParallel(self.D)
        #     self.generator = nn.DataParallel(self.G)

        # load model
        if self.args.model_path:
            checkpoint = torch.load(self.args.model_path)  # manually given a model path
        else:
            checkpoint = torch.load(os.path.join(self.args.save_path, self.save_model))

        self.D.load_state_dict(checkpoint['discriminator'])
        self.G.load_state_dict(checkpoint['generator'])
        self.D.eval()
        self.G.eval()

        check_folder(self.score_path)
        if os.path.exists(self.score_path):
            os.remove(self.score_path)

        with open(self.score_path, 'w', newline='') as f:
            f.write('Filename,NoiseType,SNR,PESQ,STOI\n')

        ''' test_file should be a noisy wav file'''
        progress_bar = tqdm(get_filepaths(self.args.WAV_test_noisy_path))
        for test_file in progress_bar:
            # modify one's test (target domain) folder structure accordingly
            noise_type = test_file.split('/')[-4]
            snr = test_file.split('/')[-3]
            speaker = test_file.split('/')[-2]
            filename = test_file.split('/')[-1]

            progress_bar.set_description(test_file, refresh=True)  # print message

            # load noisy testing WAV file
            noisy, sr = librosa.load(test_file, sr=16000)
            y, sr = librosa.load(os.path.join(self.args.WAV_test_clean_path, speaker, filename), sr=16000)

            # convert to spectrogram
            X_spec, noisy_phase, noisy_len = make_spectrum(y=noisy)
            X_spec = torch.from_numpy(X_spec.transpose()).to(self.device).unsqueeze(0)

            # model enhancement
            y_spec_pred = self.G(self.args.amplify * X_spec).detach().cpu().numpy().squeeze(0) / self.args.amplify
            y_pred = recons_spec_phase(y_spec_pred.transpose(), noisy_phase, noisy_len)

            # save enhanced WAVs
            out_path = os.path.join(self.args.WAV_enhanced_path, f"epoch{self.epochs}/")
            check_folder(out_path)
            audio_file = os.path.join(out_path, "_".join(test_file.split('/')[-4:]))
            audiowrite(audio_file, 16000, np.int16(y_pred * maxv))

            # compute scores
            score_PESQ, score_STOI = cal_score(np.int16(y * maxv), np.int16(y_pred * maxv))

            wav_name = os.path.splitext(filename)[0]
            with open(self.score_path, 'a') as f:
                f.write(f'{wav_name},{noise_type},{snr},{score_PESQ},{score_STOI}\n')