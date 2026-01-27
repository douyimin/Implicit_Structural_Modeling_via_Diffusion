import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from seisDataset.RGTDataset import RGT_dataset
from torch.utils.data import DataLoader


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / (_range + 1e-6)


model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict('/home/yimindou/Modelling/epoch=8-step=37503.ckpt', location='cpu'), strict=1)
model = model.cuda()
ddim_sampler = DDIMSampler(model)

org_dataset = RGT_dataset(r'../3DRGTData')
#org_dataset = RGT_dataset(r'F:\vae_dataset\RGT_DATA\train256x256x256')

org_dataloader = DataLoader(org_dataset)

for i, (rgt, fault, horiz) in enumerate(org_dataloader):
    with torch.no_grad():
        show_fault = fault.numpy()[0]
        show_horzi = horiz.numpy()[0]
        rgt = rgt.numpy()[0]

        fault = fault[0].permute(1, 0, 2, 3).float().cuda()
        horiz = horiz[0].permute(1, 0, 2, 3).float().cuda()
        B, C, H, W = horiz.shape
        prompt = torch.from_numpy(np.load('clip_txt.npy')).cuda()
        prompt = torch.repeat_interleave(prompt, repeats=B, dim=0)
        # seed = random.randint(0, 65535)
        cond = {"fault": fault, "horiz": horiz, "c_crossattn": [prompt]}
        # un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt])]}

        shape = (4, H // 8, W // 8)
        # model.control_scales = [1. * (0.825 ** float(12 - i)) for i in range(13)]
        # model.control_scales = [1.] * 13
        samples, intermediates = ddim_sampler.sample(50, B,
                                                     shape, cond, verbose=False, eta=0.,
                                                     unconditional_guidance_scale=1.,
                                                     # x0 = horiz.cuda(),mask = mask.cuda(),
                                                     )
        latent = samples.cpu().numpy().transpose(1, 0, 2, 3)
        np.savez('decoder_train_data/1_' + str(i).zfill(5),
                 latent=latent.astype(np.float16),
                 rgt=rgt.astype(np.float16)[0],
                 horiz=show_horzi.astype(np.float16)[0],
                 fault=show_fault.astype(np.float16)[0]
                 )

        with torch.autocast(device_type='cuda'):
            x_samples = torch.cat([model.decode_first_stage(samples[i][None]) for i in range(B)], dim=0)
        x_samples = x_samples.permute(1, 0, 2, 3).float().cpu().numpy()


        plt.figure(figsize=(16,16))
        plt.subplot(421)
        plt.imshow(rgt[0, 10], cmap='jet')
        plt.subplot(422)
        plt.imshow(rgt[0, :, :, 10], cmap='jet')

        plt.subplot(423)
        plt.imshow(x_samples[0, 10], cmap='jet')
        plt.subplot(424)
        plt.imshow(x_samples[0, :, :, 10], cmap='jet')

        plt.subplot(425)
        plt.imshow(show_fault[0, 10], cmap='gray')
        plt.subplot(426)
        plt.imshow(show_fault[0, :, :, 10], cmap='gray')

        plt.subplot(427)
        plt.imshow(show_horzi[0, 10], cmap='jet')
        plt.subplot(428)
        plt.imshow(show_horzi[0, :, :, 10], cmap='jet')

        plt.savefig('decoderfig/' + str(i).zfill(5) + '.png')
