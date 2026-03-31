import abc
import copy
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
import einops
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .config import DiscriminatorConfig


class Discriminator(nn.Module, abc.ABC):
    def __init__(self, config:DiscriminatorConfig, feature_dim: int):
        super().__init__()

        self.config = config
        self.feature_dim: int = feature_dim

        self.feature_fusion: bool = config.feature_fusion

        if self.feature_fusion:
            self.num_tokens: int = config.num_tokens
            self.fused_feature_dim: int = config.fused_feature_dim if config.fused_feature_dim else self.feature_dim

            self.fusion_layer: nn.Linear = nn.Linear(self.num_tokens * self.feature_dim, self.fused_feature_dim)

        self.use_momentum = config.use_momentum
        self.momentum = config.momentum
        
        self.require_z_score: bool = False
        self.require_update_stats: bool = False
        
        self.recording_loss = []

        self.info_dict_keys = None

        self.register_buffer('running_mean', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('running_std', torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("task_id", torch.tensor(-1, dtype=torch.int64))
        self.register_buffer("connected_adapter_task_id", torch.tensor(-1, dtype=torch.int64))
        self.register_buffer("connected_adapter_indices", torch.tensor(-1, dtype=torch.int64))

    @torch.no_grad
    def update_stats(self, loss: Tensor):
        # loss should be shape (B, 1) or (B,)
        assert loss.ndim <= 2

        max_batches_tracked = torch.tensor(self.config.max_batches_tracked, dtype=torch.int64)
        if self.num_batches_tracked < max_batches_tracked and loss.numel() > 1:
            if self.use_momentum:
                self.running_mean = self.momentum * loss.mean() + (1- self.momentum) * self.running_mean
                self.running_std = self.momentum * loss.std() + (1- self.momentum) * self.running_std
                self.num_batches_tracked += torch.tensor(1, dtype=torch.int64)
            else:
                self.recording_loss.append(loss.mean())
                losses_tensor = torch.stack(self.recording_loss)
                self.running_mean = losses_tensor.mean()

                if self.num_batches_tracked > torch.tensor(0, dtype=torch.int64) and len(self.recording_loss) > 1:
                    self.running_std = losses_tensor.std()

                self.num_batches_tracked += torch.tensor(1, dtype=torch.int64)

    @torch.no_grad
    def compute_z_score(self, mean_loss: Tensor) -> tuple[Tensor, Tensor]:
        z_score = torch.abs((mean_loss - self.running_mean) / self.running_std)
        return z_score
    
    @abc.abstractmethod
    def forward(self, x: Tensor) -> tuple[Tensor, dict]:
        raise NotImplementedError
    
    @abc.abstractmethod
    def unfreeze(self) -> list:
        raise NotImplementedError



@DiscriminatorConfig.register_subclass('autoencoder')
@dataclass
class AutoencoderConfig(DiscriminatorConfig):
    hidden_dim: int = None
    latent_dim: int = None
    use_lora: bool = False
    lora_rank: int = 32
    lora_alpha: int = 32


class AutoEncoder(Discriminator):
    config_class: AutoencoderConfig
    name: str = "autoencoder"

    def __init__(self, config: AutoencoderConfig, feature_dim: int):
        super().__init__(config, feature_dim)

        if self.feature_fusion:
            input_feature_dim = self.fused_feature_dim
        else:
            input_feature_dim = feature_dim

        self.use_lora = config.use_lora

        if self.use_lora:
            # LoRA parameters
            rank = config.lora_rank
            alpha = config.lora_alpha
            self.scaling = alpha / rank

            # Encoder (LoRA style)
            self.encoder_down_A = nn.Linear(input_feature_dim, rank, bias=False)
            self.encoder_down_B = nn.Linear(rank, config.hidden_dim, bias=True)
            self.encoder_activation = nn.ReLU()
            self.encoder_up_A = nn.Linear(config.hidden_dim, rank, bias=False)
            self.encoder_up_B = nn.Linear(rank, config.latent_dim, bias=True)

            # Decoder (LoRA style)
            self.decoder_down_A = nn.Linear(config.latent_dim, rank, bias=False)
            self.decoder_down_B = nn.Linear(rank, config.hidden_dim, bias=True)
            self.decoder_activation = nn.ReLU()
            self.decoder_up_A = nn.Linear(config.hidden_dim, rank, bias=False)
            self.decoder_up_B = nn.Linear(rank, input_feature_dim, bias=True)

        else:
            # Normal Encoder
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.latent_dim),
            )

            # Normal Decoder
            self.decoder = nn.Sequential(
                nn.Linear(config.latent_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, input_feature_dim),
            )

    def encode(self, x: Tensor) -> Tensor:
        if self.use_lora:
            x = self.encoder_down_B(self.encoder_down_A(x)) * self.scaling
            x = self.encoder_activation(x)
            x = self.encoder_up_B(self.encoder_up_A(x)) * self.scaling
            return x
        else:
            return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        if self.use_lora:
            z = self.decoder_down_B(self.decoder_down_A(z)) * self.scaling
            z = self.decoder_activation(z)
            z = self.decoder_up_B(self.decoder_up_A(z)) * self.scaling
            return z
        else:
            return self.decoder(z)

    def forward(self, x: Tensor) -> tuple[Tensor, dict]:

        if self.feature_fusion:
            if x.ndim == 2:
                expanded_feature = x.unsqueeze(-1) # (B, D) -> (B, 1, D)
            else:
                expanded_feature = x

            if self.config.batch_first:
                flattened_feature = einops.rearrange(expanded_feature, "b t d ... -> b (t d) ...")
            else:
                flattened_feature = einops.rearrange(expanded_feature, "t b d ... -> b (t d) ...")
                 
            input_feature = self.fusion_layer(flattened_feature)
        else:
            input_feature = x

        latent = self.encode(input_feature)
        reconstruction = self.decode(latent)

        reconstruction_loss = F.mse_loss(reconstruction, input_feature, reduction="none")

        info_dict = {
            "reconstruction": reconstruction,
            "loss": reconstruction_loss,
            "latent": latent
        }

        if self.feature_fusion or reconstruction_loss.ndim < 3:
            mean_loss = reconstruction_loss.mean(dim=(-1)) # (B, D) -> (B,)
        else:
            if self.config.batch_first:
                mean_loss = reconstruction_loss.mean(dim=(-2, -1)) # (B, T, D) -> (B,)
            else:
                mean_loss = reconstruction_loss.mean(dim=(-3, -1)) # (T, B, D) -> (B,)

        if self.require_z_score:
            z_score = self.compute_z_score(mean_loss)
            info_dict["z_score"] = z_score
        
        if self.require_update_stats and self.training:
            self.update_stats(mean_loss)
        
        info_dict["running_mean"] = self.running_mean
        info_dict["running_std"] = self.running_std
        info_dict["num_batches_tracked"] = self.num_batches_tracked

        self.info_dict_keys = info_dict.keys()

        return mean_loss, info_dict
    
    def unfreeze(self) -> list:
        training_parameters = []
        for parameter in self.parameters():
            parameter.requires_grad = True
            training_parameters.append(parameter)

        return training_parameters


@DiscriminatorConfig.register_subclass('autoencoder_small')
@dataclass
class AutoencoderSmallConfig(DiscriminatorConfig):
    hidden_dim: int = None


class AutoEncoderSmall(Discriminator):
    config_class: AutoencoderSmallConfig
    name: str = "autoencoder_small"

    def __init__(self, config: AutoencoderSmallConfig, feature_dim: int):
        super().__init__(config, feature_dim)

        if self.feature_fusion:
            input_feature_dim = self.fused_feature_dim
        else:
            input_feature_dim = feature_dim

        
        self.encoder = nn.Linear(input_feature_dim, config.hidden_dim)

        self.activation = nn.ReLU()

        self.decoder = nn.Linear(config.hidden_dim, input_feature_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, dict]:

        if self.feature_fusion:
            if x.ndim == 2:
                expanded_feature = x.unsqueeze(-1) # (B, D) -> (B, 1, D)
            else:
                expanded_feature = x

            if self.config.batch_first:
                flattened_feature = einops.rearrange(expanded_feature, "b t d ... -> b (t d) ...")
            else:
                flattened_feature = einops.rearrange(expanded_feature, "t b d ... -> b (t d) ...")
                 
            input_feature = self.fusion_layer(flattened_feature)
        else:
            input_feature = x

        latent = self.activation(self.encoder(input_feature))
        reconstruction = self.decoder(latent)

        reconstruction_loss = F.mse_loss(reconstruction, input_feature, reduction="none")

        info_dict = {
            "reconstruction": reconstruction,
            "loss": reconstruction_loss,
            "latent": latent
        }

        if self.feature_fusion or reconstruction_loss.ndim < 3:
            mean_loss = reconstruction_loss.mean(dim=(-1)) # (B, D) -> (B,)
        else:
            if self.config.batch_first:
                mean_loss = reconstruction_loss.mean(dim=(-2, -1)) # (B, T, D) -> (B,)
            else:
                mean_loss = reconstruction_loss.mean(dim=(-3, -1)) # (T, B, D) -> (B,)

        if self.require_z_score:
            z_score = self.compute_z_score(mean_loss)
            info_dict["z_score"] = z_score
        
        if self.require_update_stats and self.training:
            self.update_stats(mean_loss)
        
        info_dict["running_mean"] = self.running_mean
        info_dict["running_std"] = self.running_std
        info_dict["num_batches_tracked"] = self.num_batches_tracked

        self.info_dict_keys = info_dict.keys()

        return mean_loss, info_dict
    
    def unfreeze(self) -> list:
        training_parameters = []
        for parameter in self.parameters():
            parameter.requires_grad = True
            training_parameters.append(parameter)

        return training_parameters


class BatchedAutoEncoderSmall(nn.Module):

    def __init__(self, config: AutoencoderSmallConfig, autoencoders: nn.ModuleList):
        super().__init__()

        self.config = config

        self.feature_fusion: bool = config.feature_fusion

        self.autoencoders: nn.ModuleList[AutoEncoderSmall] = autoencoders
        self.num_autoencoders = len(self.autoencoders)

        # Extract weights and bias
        encoder_weights = []
        encoder_bias = []
        decoder_weights = []
        decoder_bias = []

        if self.feature_fusion:
            fusion_layer_weights = []
            fusion_layer_bias = []
        
        for autoencoder in self.autoencoders:
            encoder_weights.append(copy.deepcopy(autoencoder.encoder.weight.t()))
            encoder_bias.append(copy.deepcopy(autoencoder.encoder.bias))
            decoder_weights.append(copy.deepcopy(autoencoder.decoder.weight.t()))
            decoder_bias.append(copy.deepcopy(autoencoder.decoder.bias))

            if self.feature_fusion:
                fusion_layer_weights.append(autoencoder.fusion_layer.weight.t())
                fusion_layer_bias.append(autoencoder.fusion_layer.bias)
            
            # autoencoder.cpu()

        # Register as parameters
        self.encoder_weights = nn.Parameter(torch.stack(encoder_weights, dim=0), requires_grad=False) # (N, D, H)
        self.encoder_bias = nn.Parameter(torch.stack(encoder_bias, dim=0), requires_grad=False) # (N, H)
        self.decoder_weights = nn.Parameter(torch.stack(decoder_weights, dim=0), requires_grad=False) # (N, H, D)
        self.decoder_bias = nn.Parameter(torch.stack(decoder_bias, dim=0), requires_grad=False) # (N, D)
        if self.feature_fusion:
            self.fusion_layer_weights = nn.Parameter(torch.stack(fusion_layer_weights, dim=0), requires_grad=False) # (N, T*D, F)
            self.fusion_layer_bias = nn.Parameter(torch.stack(fusion_layer_bias, dim=0), requires_grad=False) # (N, F) 

    def forward(self, x: Tensor) -> tuple[Tensor, dict]:
        if x.ndim == 2:
            expanded_feature = x.unsqueeze(-1) # (B, D) -> (B, 1, D)
        else:
            if not self.config.batch_first:
                expanded_feature = einops.rearrange(x, "t b d ... -> b t d ...") # (B, T, D) 
            else:
                expanded_feature = x # (B, T, D)
        
        if self.feature_fusion:
            
            if self.config.batch_first:
                flattened_feature = einops.rearrange(expanded_feature, "b t d ... -> b (t d) ...") # (B, T*D)
            else:
                flattened_feature = einops.rearrange(expanded_feature, "t b d ... -> b (t d) ...") # (B, T*D)
                
            input_feature = torch.einsum("bd, ndf -> nbf", flattened_feature, self.fusion_layer_weights) + self.fusion_layer_bias # (N, B, F), regard F as D -> (N, B, D)
            input_feature = input_feature.unsqueeze(-1) # (N, B, D) -> (N, B, 1, D)

            latents = torch.einsum("nbtd, ndh -> nbth", input_feature, self.encoder_weights) + self.encoder_bias[:, None, None, :] # (N, B, T, H)
            latents = torch.relu(latents) # (N, B, T, H)
        else:
            input_feature = expanded_feature # (B, T, D)

            latents = torch.einsum("btd, ndh -> nbth", input_feature, self.encoder_weights) + self.encoder_bias[:, None, None, :] # (N, B, T, H)
            latents = torch.relu(latents) # (N, B, T, H)
        
        reconstructions = torch.einsum("nbth, nhd -> nbtd", latents, self.decoder_weights) + self.decoder_bias[:, None, None, :] # (N, B, T, D)
        reconstruction_losses = (reconstructions - input_feature).pow(2) # MSE loss (N, B, T, D)

        mean_losses = reconstruction_losses.mean(dim=(-2, -1)) # (N, B, T, D) -> (N, B)

        info_dicts = []
        for i, autoencoder in enumerate(self.autoencoders):
            info_dict = {
                "reconstruction": reconstructions[i],
                "loss": reconstruction_losses[i],
                "latent": latents[i]
            }

            if autoencoder.require_z_score:
                z_score = autoencoder.compute_z_score(mean_losses[i])
                info_dict["z_score"] = z_score
            
            # if autoencoder.require_update_stats and autoencoder.training:
            #     self.update_stats(mean_losses[i])
            
            info_dict["running_mean"] = autoencoder.running_mean
            info_dict["running_std"] = autoencoder.running_std
            info_dict["num_batches_tracked"] = autoencoder.num_batches_tracked

            # self.info_dict_keys = info_dict.keys()

            info_dicts.append(info_dict)

        return mean_losses, info_dicts


class BatchedAutoEncoder(nn.Module):
    """Batched parallel inference for AutoEncoder / AutoEncoderSmall discriminators.

    Stacks weight matrices of N autoencoder instances into contiguous tensors
    and runs a single einsum instead of N sequential forward calls.
    Supports both 1-layer (AutoEncoderSmall) and 2-layer (AutoEncoder) architectures.
    LoRA AutoEncoders are not supported (create_batched_discriminator returns None for those).
    """

    def __init__(self, config, autoencoders: list):
        super().__init__()

        self.config = config
        self.feature_fusion = config.feature_fusion
        first = autoencoders[0]
        # Detect architecture: AutoEncoderSmall has bare nn.Linear encoder;
        # AutoEncoder has nn.Sequential encoder (2-layer).
        self.use_bottleneck = isinstance(first, AutoEncoder)
        self._autoencoders = autoencoders

        if self.feature_fusion:
            fus_W, fus_b = [], []
            for ae in autoencoders:
                fus_W.append(copy.deepcopy(ae.fusion_layer.weight.t()))
                fus_b.append(copy.deepcopy(ae.fusion_layer.bias))
            self.fusion_layer_weights = nn.Parameter(
                torch.stack(fus_W, dim=0), requires_grad=False)  # (N, T*D, F)
            self.fusion_layer_bias = nn.Parameter(
                torch.stack(fus_b, dim=0), requires_grad=False)   # (N, F)

        if not self.use_bottleneck:
            # 1-layer (AutoEncoderSmall): bare Linear encoder/decoder
            enc_W, enc_b, dec_W, dec_b = [], [], [], []
            for ae in autoencoders:
                enc_W.append(copy.deepcopy(ae.encoder.weight.t()))
                enc_b.append(copy.deepcopy(ae.encoder.bias))
                dec_W.append(copy.deepcopy(ae.decoder.weight.t()))
                dec_b.append(copy.deepcopy(ae.decoder.bias))
            self.encoder_weights = nn.Parameter(
                torch.stack(enc_W, dim=0), requires_grad=False)   # (N, D, H)
            self.encoder_bias = nn.Parameter(
                torch.stack(enc_b, dim=0), requires_grad=False)    # (N, H)
            self.decoder_weights = nn.Parameter(
                torch.stack(dec_W, dim=0), requires_grad=False)   # (N, H, D)
            self.decoder_bias = nn.Parameter(
                torch.stack(dec_b, dim=0), requires_grad=False)    # (N, D)
        else:
            # 2-layer (AutoEncoder): Sequential([Linear(D,H), ReLU, Linear(H,Z)])
            enc_W1, enc_b1, enc_W2, enc_b2 = [], [], [], []
            dec_W1, dec_b1, dec_W2, dec_b2 = [], [], [], []
            for ae in autoencoders:
                enc_W1.append(copy.deepcopy(ae.encoder[0].weight.t()))  # (D, H)
                enc_b1.append(copy.deepcopy(ae.encoder[0].bias))         # (H,)
                enc_W2.append(copy.deepcopy(ae.encoder[2].weight.t()))  # (H, Z)
                enc_b2.append(copy.deepcopy(ae.encoder[2].bias))         # (Z,)
                dec_W1.append(copy.deepcopy(ae.decoder[0].weight.t()))  # (Z, H)
                dec_b1.append(copy.deepcopy(ae.decoder[0].bias))         # (H,)
                dec_W2.append(copy.deepcopy(ae.decoder[2].weight.t()))  # (H, D)
                dec_b2.append(copy.deepcopy(ae.decoder[2].bias))         # (D,)
            self.enc_W1 = nn.Parameter(torch.stack(enc_W1, dim=0), requires_grad=False)  # (N, D, H)
            self.enc_b1 = nn.Parameter(torch.stack(enc_b1, dim=0), requires_grad=False)  # (N, H)
            self.enc_W2 = nn.Parameter(torch.stack(enc_W2, dim=0), requires_grad=False)  # (N, H, Z)
            self.enc_b2 = nn.Parameter(torch.stack(enc_b2, dim=0), requires_grad=False)  # (N, Z)
            self.dec_W1 = nn.Parameter(torch.stack(dec_W1, dim=0), requires_grad=False)  # (N, Z, H)
            self.dec_b1 = nn.Parameter(torch.stack(dec_b1, dim=0), requires_grad=False)  # (N, H)
            self.dec_W2 = nn.Parameter(torch.stack(dec_W2, dim=0), requires_grad=False)  # (N, H, D)
            self.dec_b2 = nn.Parameter(torch.stack(dec_b2, dim=0), requires_grad=False)  # (N, D)

    def forward(self, x: Tensor) -> tuple[Tensor, list]:
        # Normalize input to (B, T, D)
        if x.ndim == 2:
            expanded = x.unsqueeze(1)  # (B, D) -> (B, 1, D)
        elif not self.config.batch_first:
            expanded = einops.rearrange(x, "t b d ... -> b t d ...")
        else:
            expanded = x  # (B, T, D)

        if self.feature_fusion:
            if self.config.batch_first:
                flat = einops.rearrange(expanded, "b t d ... -> b (t d) ...")
            else:
                flat = einops.rearrange(expanded, "t b d ... -> b (t d) ...")
            input_feature = (
                torch.einsum("bd,ndf->nbf", flat, self.fusion_layer_weights)
                + self.fusion_layer_bias[:, None, :]
            ).unsqueeze(2)  # (N, B, 1, F)
            use_n_einsum = True
        else:
            input_feature = expanded  # (B, T, D)
            use_n_einsum = False

        if not self.use_bottleneck:
            # 1-layer forward
            if use_n_einsum:
                latents = (
                    torch.einsum("nbtd,ndh->nbth", input_feature, self.encoder_weights)
                    + self.encoder_bias[:, None, None, :]
                )
            else:
                latents = (
                    torch.einsum("btd,ndh->nbth", input_feature, self.encoder_weights)
                    + self.encoder_bias[:, None, None, :]
                )
            latents = torch.relu(latents)  # (N, B, T, H)
            reconstructions = (
                torch.einsum("nbth,nhd->nbtd", latents, self.decoder_weights)
                + self.decoder_bias[:, None, None, :]
            )  # (N, B, T, D)
            info_latents = latents
        else:
            # 2-layer forward
            if use_n_einsum:
                l1 = (
                    torch.einsum("nbtd,ndh->nbth", input_feature, self.enc_W1)
                    + self.enc_b1[:, None, None, :]
                )
            else:
                l1 = (
                    torch.einsum("btd,ndh->nbth", input_feature, self.enc_W1)
                    + self.enc_b1[:, None, None, :]
                )
            l1 = torch.relu(l1)  # (N, B, T, H)
            l2 = (
                torch.einsum("nbth,nhz->nbtz", l1, self.enc_W2)
                + self.enc_b2[:, None, None, :]
            )  # (N, B, T, Z)
            r1 = (
                torch.einsum("nbtz,nzh->nbth", l2, self.dec_W1)
                + self.dec_b1[:, None, None, :]
            )
            r1 = torch.relu(r1)  # (N, B, T, H)
            reconstructions = (
                torch.einsum("nbth,nhd->nbtd", r1, self.dec_W2)
                + self.dec_b2[:, None, None, :]
            )  # (N, B, T, D)
            info_latents = l2

        reconstruction_losses = (reconstructions - input_feature).pow(2)  # (N, B, T, D)
        mean_losses = reconstruction_losses.mean(dim=(-2, -1))  # (N, B)

        info_dicts = []
        for i, ae in enumerate(self._autoencoders):
            info_dict = {
                "reconstruction": reconstructions[i],
                "loss": reconstruction_losses[i],
                "latent": info_latents[i],
            }
            if ae.require_z_score:
                info_dict["z_score"] = ae.compute_z_score(mean_losses[i])
            info_dict["running_mean"] = ae.running_mean
            info_dict["running_std"] = ae.running_std
            info_dict["num_batches_tracked"] = ae.num_batches_tracked
            info_dicts.append(info_dict)

        return mean_losses, info_dicts


def create_batched_discriminator(discriminators) -> Optional[nn.Module]:
    """Return a batched discriminator if all discriminators are the same batchable type.

    Returns None if types are mixed or the discriminator type has no batched implementation
    (e.g. VAE, RND, LoRA AutoEncoder) -- callers should fall back to sequential execution.
    """
    if not discriminators:
        return None
    disc_list = list(discriminators)
    if not all(type(d) is type(disc_list[0]) for d in disc_list):
        return None  # mixed types: sequential fallback

    first = disc_list[0]
    if isinstance(first, AutoEncoder):
        if getattr(first, 'use_lora', False):
            return None  # LoRA weights are not batchable
        return BatchedAutoEncoder(first.config, disc_list)
    if isinstance(first, AutoEncoderSmall):
        return BatchedAutoEncoder(first.config, disc_list)
    return None  # VAE, RND, unknown: no batched version


@DiscriminatorConfig.register_subclass('vae')
@dataclass
class VAEConfig(DiscriminatorConfig):
    hidden_dim: int = None
    latent_dim: int = None
    beta: float = 1.0  # KL weight


class VariationalAutoEncoder(Discriminator):
    config_class = VAEConfig
    name = "vae"

    def __init__(self, config: VAEConfig, feature_dim: int):
        super().__init__(config, feature_dim)
        h = config.hidden_dim
        z = config.latent_dim

        # Encoder: x -> hidden -> (mu, logvar)
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, h),
            nn.ReLU(),
        )
        self.enc_mu = nn.Linear(h, z)
        self.enc_logvar = nn.Linear(h, z)

        # Decoder: z -> hidden -> x_hat
        self.decoder = nn.Sequential(
            nn.Linear(z, h),
            nn.ReLU(),
            nn.Linear(h, feature_dim),
        )

        self.beta = config.beta
        self.feature_dim = feature_dim
        self.latent_dim = z

    @staticmethod
    def _reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        # std = exp(0.5 * logvar)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor) -> tuple[Tensor, dict]:
        if self.feature_fusion:
            if x.ndim == 2:
                expanded_feature = x.unsqueeze(-1) # (B, D) -> (B, 1, D)
            else:
                expanded_feature = x

            if self.config.batch_first:
                flattened_feature = einops.rearrange(expanded_feature, "b t d ... -> b (t d) ...")
            else:
                flattened_feature = einops.rearrange(expanded_feature, "t b d ... -> b (t d) ...")
                 
            input_feature = self.fusion_layer(flattened_feature)
        else:
            input_feature = x

        h = self.encoder(input_feature)
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)

        # Reparameterization trick
        if self.training:
            z = self._reparameterize(mu, logvar)
        else:
            # use mean at eval for deterministic recon
            z = mu

        reconstruction = self.decoder(z)

        # Per-element reconstruction loss (same as AE)
        recon_loss = F.mse_loss(reconstruction, input_feature, reduction="none")

        # KL divergence per sample: shape [batch]
        # KL(N(mu, sigma) || N(0, I)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar), dim=1)

        # Make KL term broadcastable to per-feature loss: distribute evenly across features
        kl_expanded = (self.beta * kl_loss).unsqueeze(1) / input_feature.size(1)

        total_loss = recon_loss + kl_expanded

        info_dict = {
            "reconstruction": reconstruction,
            "loss": total_loss,
            "mu": mu,
            "logvar": logvar,
            "latent": z,
            "kl_loss": kl_loss
        }

        if self.feature_fusion or total_loss.ndim < 3:
            mean_loss = total_loss.mean(dim=(-1)) # (B, D) -> (B,)
        else:
            if self.config.batch_first:
                mean_loss = total_loss.mean(dim=(-2, -1)) # (B, T, D) -> (B,)
            else:
                mean_loss = total_loss.mean(dim=(-3, -1)) # (T, B, D) -> (B,)

        if self.require_z_score:
            z_score = self.compute_z_score(mean_loss)
            info_dict["z_score"] = z_score
        
        if self.require_update_stats and self.training:
            self.update_stats(mean_loss)
        
        info_dict["running_mean"] = self.running_mean
        info_dict["running_std"] = self.running_std
        info_dict["num_batches_tracked"] = self.num_batches_tracked

        self.info_dict_keys = info_dict.keys()

        return mean_loss, info_dict

    def unfreeze(self) -> list:
        training_parameters = []
        for p in self.parameters():
            p.requires_grad = True
            training_parameters.append(p)
        return training_parameters


@dataclass
class RNDConfig(DiscriminatorConfig):
    # MLP sizes
    hidden_dim: int = None          # predictor hidden size
    target_hidden_dim: int = None   # target hidden size (defaults to hidden_dim if None)
    embedding_dim: int = None       # output feature size of both nets

    # Scale to weight the RND term relative to other discriminators
    beta: float = 1.0


class RNDDiscriminator(Discriminator):
    config_class = RNDConfig
    name = "rnd"

    def __init__(self, config: RNDConfig, feature_dim: int):
        super().__init__(config, feature_dim)

        h_pred = config.hidden_dim
        h_tgt = config.target_hidden_dim or h_pred
        d_emb = config.embedding_dim

        # Target network: randomly initialized, never trained
        self.target = nn.Sequential(
            nn.Linear(feature_dim, h_tgt),
            nn.ReLU(),
            nn.Linear(h_tgt, d_emb),
        )
        for p in self.target.parameters():
            p.requires_grad = False
        self.target.eval()  # remain in eval; no dropout/bn anyway but explicit

        # Predictor network: trained to match target features
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, h_pred),
            nn.ReLU(),
            nn.Linear(h_pred, d_emb),
        )

        self.beta = config.beta
        self.feature_dim = feature_dim
        self.embedding_dim = d_emb

    def forward(self, x: Tensor) -> tuple[Tensor, dict]:
        with torch.no_grad():
            t_feat = self.target(x)

        p_feat = self.predictor(x)

        # RND error per element in embedding space [B, d_emb]
        rnd_err = F.mse_loss(p_feat, t_feat, reduction="none")  # [B, d_emb]

        # Reduce to per-sample scalar [B] (mean over embedding)
        rnd_per_sample = rnd_err.mean(dim=1)

        # Broadcast to per-feature like AE: [B, F]
        rnd_per_feature = (self.beta * rnd_per_sample).unsqueeze(1) / x.size(1)

        info_dict = {
            "target_features": t_feat,
            "predictor_features": p_feat,
            "rnd_per_sample": rnd_per_sample,
        }
        
        return rnd_per_feature, info_dict

    def train(self, mode: bool = True):
        # keep target frozen & in eval regardless of outer mode
        super().train(mode)
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad = False
        return self

    def unfreeze(self) -> list:
        # Only predictor should train
        params = []
        for p in self.predictor.parameters():
            p.requires_grad = True
            params.append(p)
        # ensure target stays frozen
        for p in self.target.parameters():
            p.requires_grad = False
        return params


def get_discriminaor_class(name: str) -> Discriminator:
    """Get the discriminaor's class and config class given a name (matching the discriminaor class' `name` attribute)."""
    if name == "autoencoder":
        return AutoEncoder
    elif name == "autoencoder_small":
        return AutoEncoderSmall
    elif name == "vae":
        return VariationalAutoEncoder
    else:
        raise NotImplementedError(f"Discriminator with name {name} is not implemented.")

