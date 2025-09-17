from trainers.base_trainer_accelerate import BaseTrainer
from easydict import EasyDict
import torch
from local_datasets.base.base_dataset import sample_resolutions
import hydra

from pi3.models.loss import Pi3Loss
from Pi3_LoRA_helper import build_pi3_with_lora

class Pi3TrainerLoRA(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.train_loss = hydra.utils.instantiate(cfg.loss.train_loss)
        self.test_loss = hydra.utils.instantiate(cfg.loss.train_loss)
    
    
    def prepare_model(self):
        """override to load a whitened Pi3 and wrap with LoRA"""
        # Read LoRA config from Hydra cfg
        phase = self.cfg.lora.phase   # "U", "V", or "NONE"
        ckpt  = self.cfg.model.ckpt   # path to your whitened/merged checkpoint
        r     = self.cfg.lora.r
        alpha = self.cfg.lora.alpha
        drop  = self.cfg.lora.dropout

        model, _ = build_pi3_with_lora(
            ckpt_path=ckpt, device=self.accelerator.device,
            phase=phase, r=r, alpha=alpha, dropout=drop
        )
        model.to(self.accelerator.device)
        print("✅Loaded Pi3 with LoRA U config in the LoRA trainer✅")


        # set training dtype if you do mixed precision later via accelerate
        if self.cfg.train.model_dtype == "fp16":
            model.half()

        # (optional) print trainable counts with PEFT
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

        return model

    def build_optimizer(self, cfg_optimizer, model):
        def param_group_fn(model_):
            encoder_params = [param for param in model_.encoder.named_parameters()]
            other_params = [
                (name, param) for name, param in model_.named_parameters()
                if not name.startswith("encoder.") and not '.encoder.' in name
            ]

            print(f'Number of trainable encoder parameters:', sum(p.numel() for _, p in encoder_params if p.requires_grad))
            print(f'Length of trainable others:', sum(p.numel() for _, p in other_params if p.requires_grad))

            def handle_weight_decay(params, weight_decay, lr):
                decay = []
                no_decay = []
                for name, param in params:
                    if not param.requires_grad:
                        continue

                    if param.ndim <= 1 or name.endswith(".bias"):
                        no_decay.append(param)
                    else:
                        decay.append(param)

                return [
                    {"params": no_decay, "weight_decay": 0.0, 'lr': lr},
                    {"params": decay, "weight_decay": weight_decay, 'lr': lr},
                ]

            res = []
            res.extend(handle_weight_decay(encoder_params, cfg_optimizer.weight_decay, cfg_optimizer.encoder_lr))
            res.extend(handle_weight_decay(other_params, cfg_optimizer.weight_decay, cfg_optimizer.lr))

            return res
        
        return super().build_optimizer(cfg_optimizer, model, param_group_fn=param_group_fn)

    def before_epoch(self, epoch):
        if hasattr(self.train_loader, 'dataset') and hasattr(self.train_loader.dataset, 'set_epoch'):
            self.train_loader.dataset.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        if hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        if hasattr(self.train_loader, 'batch_sampler') and hasattr(self.train_loader.batch_sampler, 'batch_sampler') and hasattr(self.train_loader.batch_sampler.batch_sampler, 'sampler') and hasattr(self.train_loader.batch_sampler.batch_sampler.sampler, 'set_epoch'):       # handle acclerate warpped dataloader (more gpu)
            self.train_loader.batch_sampler.batch_sampler.sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        if hasattr(self.train_loader, 'batch_sampler') and hasattr(self.train_loader.batch_sampler, 'set_epoch'):       # handle acclerate warpped dataloader (more gpu)
            self.train_loader.batch_sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        

        if hasattr(self.test_loader, 'dataset') and hasattr(self.test_loader.dataset, 'set_epoch'):
            self.test_loader.dataset.set_epoch(0, base_seed=self.cfg.train.base_seed)
        if hasattr(self.test_loader, 'batch_sampler') and hasattr(self.test_loader.batch_sampler, 'batch_sampler') and hasattr(self.test_loader.batch_sampler.batch_sampler, 'sampler') and hasattr(self.test_loader.batch_sampler.batch_sampler.sampler, 'set_epoch'):       # handle acclerate warpped dataloader (more gpu)
            self.test_loader.batch_sampler.batch_sampler.sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)
        if hasattr(self.test_loader, 'batch_sampler') and hasattr(self.train_loader.batch_sampler, 'set_epoch'):       # handle acclerate warpped dataloader (more gpu)
            self.test_loader.batch_sampler.set_epoch(epoch, base_seed=self.cfg.train.base_seed)

        if 'random_reslution' in self.cfg.train and self.cfg.train.random_reslution and self.cfg.train.num_resolution > 0:
            seed = epoch + self.cfg.train.base_seed
            resolutions = sample_resolutions(aspect_ratio_range=self.cfg.train.aspect_ratio_range, pixel_count_range=self.cfg.train.pixel_count_range, patch_size=self.cfg.train.patch_size, num_resolutions=self.cfg.train.num_resolution, seed=seed)
            print('[Pi3 Trainer] Sampled new resolutions:', resolutions)
            datasets = []
            recursive_get_dataset(self.train_loader.dataset, datasets)
            for dataset in datasets:
                dataset._set_resolutions(resolutions)
            
    def forward_batch(self, batch, mode='train'):
        imgs = torch.stack([view['img'] for view in batch], dim=1)
        pred = self.model(imgs)

        return [pred, batch]
    
    def calculate_loss(self, output, batch, mode='train'):
        output, batch = output

        if mode == 'train':
            loss, details = self.train_loss(output, batch)
        else:
            loss, details = self.test_loss(output, batch)

        return EasyDict(
            loss=loss,
            **details
        )


def recursive_get_dataset(dataset, res=[]):
    if hasattr(dataset, 'datasets'):
        for ds in dataset.datasets:
            recursive_get_dataset(ds, res)
    else:
        if hasattr(dataset, 'dataset'):
            res.append(dataset.dataset)
    return res
