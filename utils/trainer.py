from asyncore import write
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter


class trainer():
    def __init__(self,
                 model            : nn.Module,
                 optimizer        : torch.optim.Optimizer,
                 train_dataset    : Dataset,
                 test_dataset     : Dataset,
                 epochs           : int,
                 batch_size       : int,
                 step_size        : int,
                 log_interval     : int,
                 save_dir         : str = None,
                 lr_scheduler     : torch.optim.lr_scheduler._LRScheduler = None,
                 use_amp          : bool = False,
                 *args, **kwargs) -> None:

        r'''
        Initialize the trainer class.
        
        Parameters
        --------------------------------------------------------------
        model : nn.Module
            The model to be trained.
            accuracy and loss_fn needed to be implemented.
        optimizer : torch.optim.Optimizer
            The optimizer to be used.
        train_dataset : Dataset
            The dataset for training.
        test_dataset : Dataset
            The dataset for testing.
        epochs : int
            The number of epochs to train for.
        step_size : int
            The step size for the optimization.
        log_interval : int
            The number of batches to wait before logging.
        save_dir : str
            The directory to save the model.
        lr_scheduler : torch.optim.lr_scheduler._LRScheduler
            The learning rate scheduler.
        '''

        self.model            = model
        self.optimizer        = optimizer
        if lr_scheduler is None:        
            self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, 1.0)
        else:
            self.lr_scheduler = lr_scheduler

        self.train_dataset    = train_dataset
        self.test_dataset     = test_dataset
        self.batch_size       = batch_size

        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           num_workers=4,
                                           pin_memory=True,
                                           shuffle=True)
        self.test_dataloader  = DataLoader(test_dataset,
                                           batch_size=batch_size,
                                           num_workers=4,
                                           pin_memory=True,
                                           shuffle=True)

        self.epoch            = 0
        self.epochs           = epochs
        self.step_size        = step_size
        self.log_interval     = log_interval
        self.use_amp          = use_amp

        self.scaler           = torch.cuda.amp.GradScaler(enabled = use_amp)
        self.save_dir         = save_dir
        if save_dir is not None:
            self.load()
            self.writer       = SummaryWriter(log_dir = save_dir)
            self.writer_path  = {}
            self.writer_root   = ""
        else:
            self.writer       = None
        self.device           = next(self.model.parameters()).device

        self._step = 0
        self._accu = 0.0
        self._loss = 0.0

    def cpu(self):
        r'''
        Move the model to the cpu.
        '''
        self.model  = self.model.cpu()
        self.device = next(self.model.parameters()).device
        return

    def cuda(self):
        r'''
        Move the model to the gpu.
        '''
        self.model  = self.model.cuda()
        self.device = next(self.model.parameters()).device
        return

    def _train_a_epoch(self, *args, **kwargs):
        self.model.train()
        length = len(self.train_dataloader)
        for n, batch in enumerate(self.train_dataloader):
            with torch.cuda.amp.autocast(enabled=self.use_amp) :
                inputs, targets = batch
                inputs  = inputs .to(self.device)
                targets = targets.to(self.device)

                inference = self.model(inputs)
                loss = self.model.loss_fn (inference, targets)
                accu = self.model.accuracy(inference, targets)
                self.scaler.scale(loss).backward()
                self._set_metrics(loss.item(), accu.item())
                
                _getloss, _getaccu = self._get_metrics(False)
                print("                                                                                \r",
                        'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(self.epoch, 
                        (n + 1) *  self.batch_size if (n + 1) *  self.batch_size < len(self.train_dataloader.dataset) else len(self.train_dataloader.dataset)
                        , len(self.train_dataloader.dataset), 100. * (n + 1) / len(self.train_dataloader), _getloss, _getaccu * 100), end="\r")
                        
                if n % self.step_size    == self.step_size    - 1 or n == length - 1:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                if n % self.log_interval == self.log_interval - 1 or n == length - 1:
                    _getloss, _getaccu = self._get_metrics(True)
                    print("")
                    if self.writer is not None:
                        self._add_scalar('loss',     _getloss)
                        self._add_scalar('accuracy', _getaccu * 100)
        print("")
        if self.save_dir is not None:
            self.save()
        self.lr_scheduler.step()
        return

    def _test_a_epoch(self, *args, **kwargs):
        self.model.eval()
        with torch.no_grad():
            for n, batch in enumerate(self.test_dataloader):

                inputs, targets = batch
                inputs  = inputs .to(self.device)
                targets = targets.to(self.device)

                inference = self.model(inputs)
                loss = self.model.loss_fn (inference, targets)
                accu = self.model.accuracy(inference, targets)
                self._set_metrics(loss.item(), accu.item())

                _getloss, _getaccu = self._get_metrics(False)
                print("                                                                               \r",
                    'Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(self.epoch, 
                    (n + 1) *  self.batch_size if (n + 1) *  self.batch_size < len(self.test_dataloader.dataset) else len(self.test_dataloader.dataset),
                    len(self.test_dataloader.dataset), 100. * (n + 1) / len(self.test_dataloader), _getloss, _getaccu * 100), end="\r")
            print("")
        _getloss, _getaccu = self._get_metrics()
        if self.writer is not None:
            self._add_scalar('loss',     _getloss)
            self._add_scalar('accuracy', _getaccu * 100)
        return
    

    def train(self, *args, **kwargs):
        r'''
        Train the model.
        '''
        for self.epoch in range(1, self.epochs + 1):
            self._set_writer("Train/")
            self._train_a_epoch(self.train_dataloader)
            self._set_writer("Test/")
            self._test_a_epoch (self.test_dataloader)
        return

    def save(self, *args, **kwargs):
        r'''
        Save the model.
        '''
        torch.save({
            'epoch'                  : self.epoch,
            'model_state_dict'       : self.model.state_dict(),
            'optimizer_state_dict'   : self.optimizer.state_dict(),
            'scaler_state_dict'      : self.scaler.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'loss'                   : self._loss,
            'accuracy'               : self._accu,
            'step'                   : self._step,
        }, os.path.join(self.save_dir, 'checkpoint_{}.pth'.format(self.epoch)))
        return

    def load(self, *args, **kwargs):
        r'''
        Load the model.
        '''
        try:
            for e in range(self.epochs + 1):
                load_dict = torch.load(os.path.join(self.save_dir, 'checkpoint_{}.pth'.format(e)))
        except:
            pass
        try:
            self.model.load_state_dict       (load_dict['model_state_dict'])
            self.optimizer.load_state_dict   (load_dict['optimizer_state_dict'])
            self.scaler.load_state_dict      (load_dict['scaler_state_dict'])
            self.lr_scheduler.load_state_dict(load_dict['lr_scheduler_state_dict'])
            self.epoch = load_dict['epoch']
            self._loss = load_dict['loss']
            self._accu = load_dict['accuracy']
            self._step = load_dict['step']
        except:
            pass
        return 
        
    def _set_metrics(self, loss, accu, *args, **kwargs):
        r'''
        Log the metrics of the model.
        '''
        self._step += 1
        self._loss += loss
        self._accu += accu
        return

    def _get_metrics(self, reset : bool = True, *args, **kwargs):
        r'''
        Get the metrics of the model.
        '''
        if reset:
            _l = self._loss / self._step
            _a = self._accu / self._step
            self._step = 0
            self._loss = 0.0
            self._accu = 0.0
            return _l, _a
        else :
            return self._loss / self._step, self._accu / self._step

    def _set_writer(self, root : str, *args, **kwargs):
        if root == "":
            self.writer = SummaryWriter(log_dir = self.save_dir)
        else :
            try:
                i = self.writer_path[root]
                pass
            except:
                self.writer_path[root] = 0
                pass
        self.writer_root = root
        
    def _add_scalar(self, tag : str, scalar, *args, **kwargs):
        r'''
        Set the writer of the model.
        '''
        self.writer = SummaryWriter(log_dir = self.save_dir + '/' + self.writer_root)
        self.writer.add_scalar(tag, scalar, self.writer_path[self.writer_root])
        self.writer_path[self.writer_root] += 1
        return
    