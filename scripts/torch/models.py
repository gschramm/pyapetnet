import torch
import torchio as tio
import pytorch_lightning as pl

from collections import OrderedDict

class APetNet(pl.LightningModule):
  def __init__(self, loss = torch.nn.L1Loss(), petOnly = False):
    super().__init__()

    self.loss    = loss
    self.petOnly = petOnly

    self.model   = self.seq_model()

  #---------------------------------------------
  def forward(self, x):
    return torch.nn.ReLU()(self.model(x) + x[:,0:1,...])

  #---------------------------------------------
  def training_step(self, batch, batch_idx):
    return self.step(batch, batch_idx, mode = 'training')

  #---------------------------------------------
  def validation_step(self, batch, batch_idx):
    return self.step(batch, batch_idx, mode = 'validation')

  #---------------------------------------------
  def step(self, batch, batch_idx, mode = 'training'):

    if self.petOnly:
      x  = batch['pet_low'][tio.DATA]
    else:
      x0 = batch['pet_low'][tio.DATA]
      x1 = batch['mr'][tio.DATA]
      x  = torch.cat((x0,x1),1)

    y  = batch['pet_high'][tio.DATA]

    y_hat = self.forward(x)

    loss = self.loss(y_hat, y)
    # Logging to TensorBoard by default
    if mode == 'training':
      self.log('train_loss', loss)
    elif mode == 'validation':
      self.log('val_loss', loss)
    return loss

  #---------------------------------------------
  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    return optimizer

  #---------------------------------------------
  def conv_act_block(self, in_channels, out_channels, device, kernel_size = (3,3,3), 
                     activation = None, batchnorm = True):
  
    od = OrderedDict()
  
    od['conv'] = torch.nn.Conv3d(in_channels = in_channels, out_channels = out_channels, 
                                 kernel_size = kernel_size, padding = 'same', device = self.device)
   
    if batchnorm:
      od['bnorm'] =  torch.nn.BatchNorm3d(out_channels, device = self.device)
  
    if activation is None:
      od['act'] = torch.nn.PReLU(num_parameters = 1, device = self.device)
    else:
      od['act'] = activation
  
    return torch.nn.Sequential(od)
  
  #---------------------------------------------
  def seq_model(self, nfeat = 30, nblocks = 8):
    
    od = OrderedDict()
    # add first conv layer 
    if self.petOnly:
      od['b0'] = self.conv_act_block(1, nfeat, self.device, kernel_size = (3,3,3))
    else:
      od['b0'] = self.conv_act_block(2, nfeat, self.device, kernel_size = (3,3,3))
    
    for i in range(nblocks):
      od[f'b{i+1}'] = self.conv_act_block(nfeat, nfeat, self.device, kernel_size = (3,3,3))
    
    od[f'conv111'] = torch.nn.Conv3d(in_channels = nfeat, out_channels = 1, 
                                     kernel_size = (1,1,1), padding = 'same', device = self.device)
    model = torch.nn.Sequential(od)
  
    return model
