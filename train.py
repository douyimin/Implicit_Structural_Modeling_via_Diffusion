import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
import torch
from share import *
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from struc2rgtDataset import seisDataset
from pytorch_lightning.strategies.ddp import DDPStrategy
from valDataset import valDataset
from pytorch_lightning.callbacks import ModelCheckpoint
# Configs
resume_path = '512x512.ckpt'
batch_size =25
logger_freq = 500
learning_rate = 1e-5



# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('cldm_v21.yaml').cpu()
model.load_ema_to_model(resume_path)
#msg = model.load_state_dict(load_state_dict(resume_path, location='cpu'),strict = 0)
#print(msg)
model.learning_rate = learning_rate
checkpoint_callback = ModelCheckpoint(
    every_n_train_steps=5000,    # 每 2000 个训练步保存一次 (修改这里)
    #every_n_epochs=1, # 每 1 个 epoch 保存一次
    filename='last.ckpt',
    save_last=True               # 始终保存最新的一个 last.ckpt
)

# Misc
dataset = seisDataset(r'/data3/douyimin/RGT_Seis_Fault_2D')
dataloader = DataLoader(dataset, num_workers=48, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(valDataset('val_data'), num_workers=16, batch_size=1, shuffle=1)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(precision='16-mixed', callbacks=[logger,checkpoint_callback]
                     ,strategy = DDPStrategy(static_graph=True,find_unused_parameters=True))
#trainer = pl.Trainer(precision=16, callbacks=[logger])

if __name__=='__main__':
    trainer.fit(model, dataloader,val_loader)
