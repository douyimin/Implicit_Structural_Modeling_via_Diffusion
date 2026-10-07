"""Paper training scope and checkpointed EMA of all trainable parameters."""
import torch
from pytorch_lightning import Callback

from ISM_diffusion.cldm import ControlLDM


def freeze_for_paper(model):
    model.requires_grad_(False)
    for module in (model.control_fault, model.control_horiz,
                   model.model.diffusion_model.output_blocks, model.model.diffusion_model.out):
        module.requires_grad_(True)
    model.first_stage_model.eval()
    model.sd_locked = False


class PaperControlLDM(ControlLDM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        freeze_for_paper(self)
        self.optimizer_options = dict(betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    def configure_optimizers(self):
        return torch.optim.AdamW((p for p in self.parameters() if p.requires_grad),
                                 lr=self.learning_rate, **self.optimizer_options)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss, _ = self.shared_step(batch)
        self.log("val/diffusion_loss_ema", loss, on_step=False, on_epoch=True,
                 sync_dist=True, batch_size=batch["jpg"].shape[0])


class TrainableEMA(Callback):
    """Fixed decay, updated once per optimizer step, including both branches."""
    def __init__(self, decay=0.9996):
        if not 0 < decay < 1:
            raise ValueError("EMA decay must be between zero and one")
        self.decay = decay
        self.shadow = {}
        self.last_step = 0
        self._backup = None

    @property
    def state_key(self):
        return "ISMTrainableEMA"

    def state_dict(self):
        return dict(decay=self.decay, last_step=self.last_step, shadow=self.shadow)

    def load_state_dict(self, state):
        if self.decay != state["decay"]:
            raise ValueError("EMA decay differs from the resumed checkpoint")
        self.last_step = state["last_step"]
        self.shadow = state["shadow"]

    def on_fit_start(self, trainer, pl_module):
        parameters = {n: p for n, p in pl_module.named_parameters() if p.requires_grad}
        if self.shadow and set(self.shadow) != set(parameters):
            raise ValueError("EMA parameter names differ from this model's training scope")
        self.shadow = {n: self.shadow.get(n, p.detach()).to(device=p.device, dtype=torch.float32).clone()
                       for n, p in parameters.items()}

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step <= self.last_step:
            return
        for name, parameter in pl_module.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(parameter.detach().float(), 1 - self.decay)
        self.last_step = trainer.global_step

    @torch.no_grad()
    def on_validation_start(self, trainer, pl_module):
        self._backup = {}
        for name, parameter in pl_module.named_parameters():
            if name in self.shadow:
                self._backup[name] = parameter.detach().clone()
                parameter.copy_(self.shadow[name])

    @torch.no_grad()
    def on_validation_end(self, trainer, pl_module):
        for name, parameter in pl_module.named_parameters():
            if self._backup is not None and name in self._backup:
                parameter.copy_(self._backup[name])
        self._backup = None


def checkpoint_weights(checkpoint, use_ema=False):
    """Select ordinary or explicitly requested EMA weights without silent fallback."""
    state = dict(checkpoint.get("state_dict", checkpoint))
    if use_ema:
        ema = checkpoint.get("callbacks", {}).get("ISMTrainableEMA")
        if ema is None or not ema.get("shadow"):
            raise ValueError("No release EMA state found; omit --ema for ordinary/exported weights")
        for name, value in ema["shadow"].items():
            if name not in state or value.shape != state[name].shape:
                raise ValueError(f"Incompatible EMA parameter: {name}")
            state[name] = value
    return state
