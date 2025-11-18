import pytorch_lightning as pl
from typing import List, Dict, Any
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from transformers.optimization import get_linear_schedule_with_warmup
from transformers.models.bert.modeling_bert import (
    BertAttention,
    BertEncoder,
    BertConfig,
)


# self-defined modules
class SELayer(nn.Module):
    # according to the paper: https://arxiv.org/pdf/2401.13858
    def __init__(self, bert_config, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(bert_config.hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(bert_config.hidden_size, elementwise_affine=False)

        self.adaLN_modulation = nn.Sequential(
            nn.Linear(bert_config.hidden_size, bert_config.hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(bert_config.hidden_size, 6 * bert_config.hidden_size, bias=True),
        )

        self.attn = BertAttention(bert_config, **block_kwargs)

        self.mlp = nn.Sequential(
            nn.Linear(
                bert_config.hidden_size, int(bert_config.hidden_size * mlp_ratio)
            ),
            nn.GELU(),
            nn.Dropout(bert_config.hidden_dropout_prob),
            nn.Linear(
                int(bert_config.hidden_size * mlp_ratio), bert_config.hidden_size
            ),
            nn.Dropout(bert_config.hidden_dropout_prob),
        )

        nn.init.zeros_(self.adaLN_modulation[0].weight)
        nn.init.zeros_(self.adaLN_modulation[0].bias)

    def forward(self, x, c, mask):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self._modulate(
            self.norm1(self.attn(x, mask)[0]), shift_msa, scale_msa
        )
        x = x + gate_mlp * self._modulate(self.norm2(self.mlp(x)), shift_mlp, scale_mlp)
        return x

    def _modulate(self, x, shift, scale):
        return x * (1 + scale) + shift


class GaussianFourierProjection(nn.Module):
    """
    Gaussian random features for encoding time steps.
    Built primarily for score-based models.

    Source:
    https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=YyQtV7155Nht
    """

    def __init__(self, embed_dim: int = 384, scale: float = 2 * torch.pi):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        w = torch.randn(embed_dim // 2) * scale
        assert w.requires_grad == False
        self.register_buffer("W", w)

    def forward(self, x: torch.Tensor):
        """
        takes as input the time vector and returns the time encoding
        time (x): (batch_size, )
        output  : (batch_size, embed_dim)
        """
        if x.ndim > 1:
            x = x.squeeze()
        elif x.ndim < 1:
            x = x.unsqueeze(0)
        x_proj = x[:, None] * self.W[None, :] * 2 * torch.pi
        embed = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return embed


# main model
class ConditionalBertForDiffusionBase(nn.Module):
    def __init__(self, feature_size: int, bert_config: BertConfig) -> None:
        super().__init__()
        decoder_config = deepcopy(bert_config)
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        self.timestep_projector = GaussianFourierProjection(bert_config.hidden_size)
        self.ligand_feature_proj = nn.Linear(feature_size, bert_config.hidden_size)
        self.receptor_feature_proj = nn.Linear(feature_size, bert_config.hidden_size)
        # self.timestep_emb = SELayer(bert_config)
        self.ligand_encoder = BertEncoder(bert_config)
        self.ligand_norm = nn.LayerNorm(bert_config.hidden_size)
        self.pocket_encoder = BertEncoder(bert_config)
        self.pocket_norm = nn.LayerNorm(bert_config.hidden_size)
        self.receptor_encoder = BertEncoder(bert_config)
        self.receptor_norm = nn.LayerNorm(bert_config.hidden_size)
        self.ligand_decoder = BertEncoder(decoder_config)
        self.decoder_norm = nn.LayerNorm(bert_config.hidden_size)
        self.output_proj = nn.Linear(bert_config.hidden_size, feature_size)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        def _constant_init(module, i):
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, i)
                if module.bias is not None:
                    nn.init.constant_(module.bias, i)

        self.apply(_basic_init)
        # _constant_init(self.timestep_emb.adaLN_modulation[0], 0)

    def forward(
        self,
        timestep,
        noised_ligand_emb,
        ligand_masks,
        receptor_emb,
        receptor_masks,
        pocket_mask,
    ):
        ligand_masks = self._exetend_attention_mask(ligand_masks)
        receptor_masks = self._exetend_attention_mask(receptor_masks)
        pocket_mask = self._exetend_attention_mask(pocket_mask)
        timestep_proj = self.timestep_projector(timestep.squeeze(dim=-1)).unsqueeze(1)
        noised_ligand_emb = self.ligand_feature_proj(noised_ligand_emb)  # [L, 2048]
        receptor_emb = self.receptor_feature_proj(receptor_emb)
        # Encode Features
        noised_ligand_emb = self.ligand_encoder(
            hidden_states=noised_ligand_emb, attention_mask=ligand_masks
        ).last_hidden_state  # + noised_ligand_emb + timestep_proj
        noised_ligand_emb = self.ligand_norm(noised_ligand_emb)
        noised_ligand_emb = self.timestep_emb(
            noised_ligand_emb, timestep_proj, ligand_masks
        )
        receptor_emb = self.receptor_encoder(
            hidden_states=receptor_emb, attention_mask=receptor_masks
        ).last_hidden_state  # + receptor_emb
        receptor_emb = self.receptor_norm(receptor_emb)
        pocket_emb = self.pocket_encoder(
            hidden_states=receptor_emb, attention_mask=pocket_mask
        ).last_hidden_state  # + receptor_emb
        pocket_emb = self.pocket_norm(pocket_emb)
        # Combine receptor and ligand
        denoised_ligand_emb = self.ligand_decoder(
            hidden_states=noised_ligand_emb,
            attention_mask=ligand_masks,
            encoder_hidden_states=pocket_emb,
            encoder_attention_mask=pocket_mask,
        ).last_hidden_state
        denoised_ligand_emb = self.decoder_norm(denoised_ligand_emb)
        output = self.output_proj(denoised_ligand_emb)
        return output

    def _exetend_attention_mask(self, mask):
        # From hugggingface modeling_utils
        mask = mask.float()
        extended_attention_mask = mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.type_as(mask)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


class ConditionalBertForDiffusion(ConditionalBertForDiffusionBase, pl.LightningModule):
    """
    Wraps model by pl LightningModule
    """

    def __init__(
        self,
        feature_size: int,
        bert_config: BertConfig,
        epochs: int = 1,
        lr_scheduler=None,
        l2_lambda: float = 0.0,
        steps_per_epoch: int = 250,
        learning_rate: float = 5e-5,
        **kwargs,
    ):
        """Feed args to BertForDiffusionBase and then feed the rest into"""
        ConditionalBertForDiffusionBase.__init__(self, feature_size, bert_config)
        # Store information about leraning rates and loss
        self.steps_per_epoch = steps_per_epoch
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.train_epoch_losses = []
        self.valid_epoch_losses = []
        self.epochs = epochs
        self.l2_lambda = l2_lambda
        self.train_epoch_counter = 0
        self.bert_config = bert_config
        self.loss = nn.MSELoss()
        self.cos_loss = nn.CosineEmbeddingLoss()
        self.feature_size = feature_size

    def _get_loss_terms(self, batch) -> List[torch.Tensor]:
        """
        Returns the loss terms for the model. Length of the returned list
        is equivalent to the number of features we are fitting to.
        """
        ligand_mask = batch["ligand_mask"].bool()
        known_noise = batch["noise"]

        predicted_noise = self.forward(
            timestep=batch["timestep"],
            noised_ligand_emb=batch["noised_ligand_emb"],
            ligand_masks=batch["ligand_mask"],
            receptor_emb=batch["receptor_emb"],
            receptor_masks=batch["receptor_mask"],
            pocket_mask=batch["pocket_mask"],
        )
        # print(
        #     batch["noised_ligand_emb"].shape,
        #     batch["sqrt_one_minus_alphas_cumprod_t"].shape,
        #     predicted_noise.shape,
        #     batch["sqrt_alphas_cumprod_t"].shape
        # )
        noise_scale = batch["sqrt_one_minus_alphas_cumprod_t"].view(-1, 1, 1)
        signal_scale = batch["sqrt_alphas_cumprod_t"].view(-1, 1, 1)
        predicted_emb = (
            batch["noised_ligand_emb"] - noise_scale * predicted_noise
        ) / signal_scale
        predicted_emb = predicted_emb[ligand_mask]
        ligand_emb = batch["ligand_emb"][ligand_mask]
        predicted_noise = predicted_noise[ligand_mask]
        known_noise = known_noise[ligand_mask]
        assert (
            known_noise.shape == predicted_noise.shape
        ), f"{known_noise.shape} != {predicted_noise.shape}"

        # predicted_noise = predicted_noise.reshape(-1, self.feature_size)
        # known_noise = known_noise.reshape(-1, self.feature_size)
        # target = torch.ones(known_noise.shape[0]).to(batch["noise"].device)
        # loss = self.loss(predicted_noise, known_noise, target)
        loss = (
            0.8 * self.loss(predicted_noise, known_noise)
            + 0.1 * self.loss(predicted_emb, ligand_emb)
            + 0.1
            * self.cos_loss(
                predicted_emb,
                ligand_emb,
                torch.ones(predicted_emb.shape[0], device=ligand_emb.device),
            )
        )
        return loss

    def training_step(self, batch, batch_idx):
        """
        Training step, runs once per batch
        """
        loss_terms = self._get_loss_terms(batch)
        self.log_dict(
            {"train_loss": loss_terms}, sync_dist=True
        )  # Don't seem to need rank zero or sync dist
        return loss_terms

    def on_train_batch_end(self, outputs, batch_idx, dataloader_idx=0) -> None:
        """Log the average training loss over the epoch"""
        # pl.utilities.rank_zero_info(outputs)
        self.train_epoch_losses.append(float(outputs["loss"]))

    def on_train_epoch_end(self) -> None:
        pl.utilities.rank_zero_info(
            f"Traning Loss:{sum(self.train_epoch_losses)/len(self.train_epoch_losses)}"
        )
        self.train_epoch_losses = []
        self.train_epoch_counter += 1

    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """
        Validation step
        """
        with torch.no_grad():
            loss_terms = self._get_loss_terms(batch)
        loss_dict = {"val_loss": loss_terms}
        # with rank zero it seems that we don't need to use sync_dist
        self.log_dict(loss_dict, rank_zero_only=True, sync_dist=True)
        return loss_dict

    def on_validation_epoch_end(self) -> None:
        pl.utilities.rank_zero_info(
            f"Validation Loss:{sum(self.valid_epoch_losses)/len(self.valid_epoch_losses)}"
        )
        self.valid_epoch_losses = []

    def on_validation_batch_end(self, outputs, batch_idx, dataloader_idx=0) -> None:
        """Log the average validation loss over the epoch"""
        self.valid_epoch_losses.append(float(outputs["val_loss"]))

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Return optimizer. Limited support for some optimizers
        """
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_lambda,
        )
        retval = {"optimizer": optim}
        if self.lr_scheduler:
            if self.lr_scheduler == "OneCycleLR":
                retval["lr_scheduler"] = {
                    "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                        optim,
                        max_lr=1e-2,
                        epochs=self.epochs,
                        steps_per_epoch=self.steps_per_epoch,
                    ),
                    "monitor": "val_loss",
                    "frequency": 1,
                    "interval": "step",
                }
            elif self.lr_scheduler == "LinearWarmup":
                # https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/optimizer_schedules#transformers.get_linear_schedule_with_warmup
                # Transformers typically do well with linear warmup
                warmup_steps = int(self.epochs * 0.1)
                pl.utilities.rank_zero_info(
                    f"Using linear warmup with {warmup_steps}/{self.epochs} warmup steps"
                )
                retval["lr_scheduler"] = {
                    "scheduler": get_linear_schedule_with_warmup(
                        optim,
                        num_warmup_steps=warmup_steps,
                        num_training_steps=self.epochs,
                    ),
                    "frequency": 1,
                    "interval": "epoch",  # Call after 1 epoch
                }
            else:
                raise ValueError(f"Unknown lr scheduler {self.lr_scheduler}")
        pl.utilities.rank_zero_info(f"Using optimizer {retval}")
        return retval
