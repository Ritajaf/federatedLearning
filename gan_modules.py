import copy
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from sklearn.cluster import KMeans


# CHANGE 8: Learned prototype bank (replaces KMeans prototype bank for training/conditioning)
class LearnedPrototypeBank(nn.Module):
    """
    Learnable prototype bank with cosine-similarity lookup.

    Given semantic embeddings [B, d_model], it returns the nearest prototype
    vector [B, d_model] based on cosine similarity.
    """

    def __init__(self, n_prototypes: int, d_model: int):
        super().__init__()
        if n_prototypes < 1:
            raise ValueError("n_prototypes must be >= 1")
        self.n_prototypes = int(n_prototypes)
        self.d_model = int(d_model)
        self.prototypes = nn.Parameter(torch.randn(self.n_prototypes, self.d_model))

    def forward(self, semantic_emb: torch.Tensor) -> torch.Tensor:
        # semantic_emb: [B, D] -> nearest prototype [B, D]
        emb_norm = F.normalize(semantic_emb, dim=-1)  # [B, D]
        proto_norm = F.normalize(self.prototypes, dim=-1)  # [P, D]
        sim = emb_norm @ proto_norm.t()  # [B, P]
        idx = sim.argmax(dim=-1)  # [B]
        return self.prototypes[idx]  # [B, D]


class SemanticGenerator(nn.Module):
    """
    Token-level conditional generator.

    Takes:
      - enc_output: [B, T, d_model]
      - channel_state: [B, 1]
      - prototype: [B, d_model]

    Returns:
      - correction: [B, T, d_model]  (one correction per token)

    Inputs:
        enc_output:    [B, T, d_model]
        channel_state: [B, 1]
        prototype_vec: [B, d_model]
    Output:
        [B, T, output_dim] (usually same as d_model)
    """

    def __init__(self, semantic_dim: int, channel_dim: int, prototype_dim: int, output_dim: int):
        super().__init__()
        # CHANGE 2: Token-level generator conditioning
        # Per-token input is: [enc_output (d_model) + snr (1) + prototype (d_model)]
        token_in_dim = semantic_dim + channel_dim + prototype_dim  # typically d_model + 1 + d_model = 2*d_model + 1
        attn_embed_dim = 128
        num_heads = 4

        self.token_encoder = nn.Sequential(
            nn.Linear(token_in_dim, 256),  # spec: Linear(257 -> 256) for d_model=128
            nn.ReLU(inplace=True),
            nn.Linear(256, attn_embed_dim),  # spec: Linear(256 -> 128)
            nn.ReLU(inplace=True),
        )
        self.cross_token_attn = nn.MultiheadAttention(
            embed_dim=attn_embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.out_proj = nn.Linear(attn_embed_dim, output_dim)

    def forward(
        self,
        enc_output: torch.Tensor,  # [B, T, d_model]
        channel_state: torch.Tensor,  # [B, 1]
        prototype_vec: torch.Tensor,  # [B, d_model]
    ) -> torch.Tensor:
        # Expand SNR and prototype to per-token representations
        B, T, D = enc_output.shape
        snr_per_token = channel_state.unsqueeze(1).expand(B, T, -1)  # [B, T, 1]
        proto_per_token = prototype_vec.unsqueeze(1).expand(B, T, -1)  # [B, T, d_model]

        x = torch.cat([enc_output, snr_per_token, proto_per_token], dim=-1)  # [B, T, 2*d_model+1]
        token_features = self.token_encoder(x)  # [B, T, 128]
        attn_out, _ = self.cross_token_attn(token_features, token_features, token_features, need_weights=False)
        correction = self.out_proj(attn_out)  # [B, T, d_model]
        return correction


class SemanticDiscriminator(nn.Module):
    """
    Discriminator that judges whether a reconstructed embedding is
    consistent with the original semantic embedding.

    Inputs:
        original_emb:      [batch, embedding_dim]
        reconstructed_emb: [batch, embedding_dim]
    Output:
        [batch, 1] probability in (0, 1)
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        # CHANGE 4: Spectral normalization on both linear layers
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(embedding_dim * 2, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(256, 1)),
            nn.Sigmoid(),
        )

    def forward(self, original_emb: torch.Tensor, reconstructed_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([original_emb, reconstructed_emb], dim=-1)
        return self.net(x)


def client_loss(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    disc_output: torch.Tensor,
    global_prototype: torch.Tensor,
    local_prototype: torch.Tensor,
    lambda_adv: float = 0.1,
    lambda_div: float = 0.01,
) -> torch.Tensor:
    """
    Combined client-side loss:
      1) semantic reconstruction (MSE in embedding space)
      2) adversarial loss against discriminator
      3) semantic divergence regularizer to keep local close to global prototype
    """
    # 1. semantic reconstruction of embeddings
    semantic_loss = F.mse_loss(reconstructed, original)

    # 2. adversarial loss (encourage discriminator to output 1 for fakes)
    eps = 1e-8
    adversarial_loss = -(torch.log(disc_output + eps)).mean()

    # 3. divergence between local and global prototypes
    # CHANGE 8: if no prototype bank exists yet, lambda_div_effective should be 0
    # and we should skip divergence computation to avoid NaNs from cosine similarity of zeros.
    if lambda_div == 0.0:
        divergence_loss = semantic_loss.new_tensor(0.0)
    else:
        target = torch.ones(local_prototype.size(0), device=local_prototype.device)
        divergence_loss = F.cosine_embedding_loss(local_prototype, global_prototype, target)

    return semantic_loss + lambda_adv * adversarial_loss + lambda_div * divergence_loss


def build_prototype_bank(semantic_embeddings: torch.Tensor, n_prototypes: int = 10) -> torch.Tensor:
    """
    Build a lightweight prototype bank from semantic embeddings using KMeans.

    Args:
        semantic_embeddings: [N, dim] tensor of sentence-level embeddings.
    Returns:
        prototypes: [n_prototypes, dim] tensor on the same device.
    """
    if semantic_embeddings.ndim != 2:
        raise ValueError(f"semantic_embeddings must be 2D [N, dim], got {semantic_embeddings.shape}")

    device = semantic_embeddings.device
    embeddings_np = semantic_embeddings.detach().cpu().numpy()
    n_samples = embeddings_np.shape[0]
    k = min(n_prototypes, n_samples)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(embeddings_np)
    prototypes = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
    return prototypes


def get_nearest_prototype(semantic_emb: torch.Tensor, global_prototypes: torch.Tensor) -> torch.Tensor:
    """
    For each semantic embedding, find the nearest (cosine) prototype from the global bank.

    Args:
        semantic_emb:     [batch, dim]
        global_prototypes:[K, dim]
    Returns:
        prototype_vec:    [batch, dim]
    """
    if global_prototypes is None or global_prototypes.numel() == 0:
        # fall back to zeros if no global prototypes yet
        return torch.zeros_like(semantic_emb)

    # cosine similarity between each embedding and all prototypes
    # semantic_emb: [B, D] -> [B, 1, D]
    # prototypes:   [K, D] -> [1, K, D]
    emb = F.normalize(semantic_emb, dim=-1).unsqueeze(1)
    prot = F.normalize(global_prototypes, dim=-1).unsqueeze(0)
    sim = (emb * prot).sum(-1)  # [B, K]
    idx = sim.argmax(dim=-1)    # [B]
    prototype_vec = global_prototypes[idx]
    return prototype_vec


def extract_sentence_embeddings(
    model: nn.Module,
    src_batch: torch.Tensor,
    pad_idx: int,
) -> torch.Tensor:
    """
    Run DeepSC encoder and mean-pool over time to obtain sentence-level embeddings.
    """
    device = next(model.parameters()).device

    src = src_batch.to(device)
    # simple source mask (no target here)
    src_mask = (src == pad_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)
    with torch.no_grad():
        enc_output = model.encoder(src, src_mask)  # [B, T, d_model]
    # mean-pool over sequence length
    sent_emb = enc_output.mean(dim=1)  # [B, d_model]
    return sent_emb


def local_train_gan(
    client_model: nn.Module,
    generator: nn.Module,
    discriminator: nn.Module,
    dataloader,
    global_prototype_bank: nn.Module,
    pad_idx: int,
    snr: float,
    n_disc_steps: int = 5,
    gen_lr: float = 1e-4,
    disc_lr: float = 1e-4,
    lambda_adv: float = 0.1,
    lambda_div: float = 0.01,
    n_prototypes: int = 10,
    min_gan_steps: int = 30,
) -> Tuple[dict, float, torch.Tensor, float]:
    """
    Two-timescale local training for conditional GAN on top of the DeepSC encoder.

    This does NOT change the main DeepSC training loop; it only trains the
    generator/discriminator in semantic space and returns:
        - updated generator state_dict (for federated aggregation)
        - client_snr (the scalar SNR used during GAN training)
        - local learned prototype bank parameters (if global_prototype_bank is provided)
        - local generator reconstruction loss (semantic space) for aggregation quality
    """
    device = next(client_model.parameters()).device
    client_model.eval()  # encoder is used as a fixed feature extractor here
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # CHANGE 8: Prototype bank is learned and updated during generator training (stage 2)
    local_prototype_bank = None
    if global_prototype_bank is not None:
        local_prototype_bank = copy.deepcopy(global_prototype_bank).to(device)
        gen_params = list(generator.parameters()) + list(local_prototype_bank.parameters())
    else:
        gen_params = list(generator.parameters())

    # Use Adam on generator (+ prototype params if available)
    gen_optimizer = torch.optim.Adam(gen_params, lr=gen_lr, betas=(0.9, 0.98), eps=1e-8)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=disc_lr, betas=(0.9, 0.98), eps=1e-8)

    channel_state_value = float(snr)
    gen_recon_loss_sum = 0.0
    gen_steps = 0

    # CHANGE 6: Minimum GAN steps per client (cycle through dataloader)
    batch_iter = iter(dataloader)
    while gen_steps < min_gan_steps:
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(dataloader)
            batch = next(batch_iter)

        src = batch.to(device)

        # Encoder used as fixed feature extractor
        with torch.no_grad():
            src_mask = (src == pad_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)
            enc_output = client_model.encoder(src, src_mask)  # [B, T, d_model]
            semantic_emb = enc_output.mean(dim=1)  # [B, d_model]

        B, d_model = semantic_emb.shape
        channel_state = torch.full((B, 1), channel_state_value, device=device)

        # Prototype conditioning (learned bank in stage 2, otherwise zeros)
        if local_prototype_bank is not None:
            prototype_vec_local = local_prototype_bank(semantic_emb)  # [B, d_model]
            with torch.no_grad():
                prototype_vec_global = global_prototype_bank(semantic_emb).detach()  # [B, d_model]
            lambda_div_effective = lambda_div
        else:
            prototype_vec_local = torch.zeros_like(semantic_emb)
            prototype_vec_global = torch.zeros_like(semantic_emb)
            lambda_div_effective = 0.0  # avoid meaningless cosine loss when no prototype bank exists

        # -------- Train discriminator (inner loop) --------
        for _ in range(n_disc_steps):
            with torch.no_grad():
                correction = generator(enc_output, channel_state, prototype_vec_local)  # [B, T, d_model]
                reconstructed_enc = enc_output + correction
                fake_emb = reconstructed_enc.mean(dim=1)  # [B, d_model]

            real_score = discriminator(semantic_emb, semantic_emb)
            fake_score = discriminator(semantic_emb, fake_emb)

            eps = 1e-8
            disc_loss = -torch.log(real_score + eps) - torch.log(1.0 - fake_score + eps)
            disc_loss = disc_loss.mean()

            disc_optimizer.zero_grad()
            disc_loss.backward()
            # CHANGE 3: Gradient clipping for discriminator
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            disc_optimizer.step()

        # -------- Train generator (outer step) --------
        correction = generator(enc_output, channel_state, prototype_vec_local)  # [B, T, d_model]
        reconstructed_enc = enc_output + correction
        fake_emb = reconstructed_enc.mean(dim=1)  # [B, d_model]

        fake_score = discriminator(semantic_emb, fake_emb)
        semantic_recon_loss = F.mse_loss(fake_emb, semantic_emb)

        gen_loss = client_loss(
            original=semantic_emb,
            reconstructed=fake_emb,
            disc_output=fake_score,
            global_prototype=prototype_vec_global,
            local_prototype=prototype_vec_local,
            lambda_adv=lambda_adv,
            lambda_div=lambda_div_effective,
        )

        gen_optimizer.zero_grad()
        gen_loss.backward()
        # CHANGE 3: Gradient clipping for generator (and prototype params if present)
        torch.nn.utils.clip_grad_norm_(gen_params, max_norm=1.0)
        gen_optimizer.step()

        gen_recon_loss_sum += float(semantic_recon_loss.item())
        gen_steps += 1

    # Return local prototype bank parameters for federation (stage 2)
    if local_prototype_bank is not None:
        local_prototypes_out = local_prototype_bank.prototypes.detach()
    else:
        local_prototypes_out = torch.empty(0, device=device)

    local_gen_recon_loss = gen_recon_loss_sum / max(1, gen_steps)
    return copy.deepcopy(generator.state_dict()), channel_state_value, local_prototypes_out, local_gen_recon_loss


def federated_aggregate_generator(
    client_generator_weights: List[dict],
    client_snrs: List[float],
    client_prototypes: List[torch.Tensor],
    client_gen_recon_losses: List[float],
) -> Tuple[dict, torch.Tensor]:
    """
    Federated aggregation with generator quality and prototype similarity.

    CHANGE 7:
      w_k ∝ s_SNR_k × s_proto_k × gen_quality_k
      gen_quality_k = 1 / (1 + local_gen_loss_k)
    CHANGE 8:
      prototype bank parameters are aggregated with the same weights.
    """
    n_clients = len(client_generator_weights)
    if n_clients == 0:
        raise ValueError("client_generator_weights is empty.")

    device = next(iter(client_generator_weights[0].values())).device

    # mean SNR across clients
    mean_snr = float(sum(client_snrs) / len(client_snrs))

    # SNR similarity term: closer to mean SNR is better
    snr_sim = torch.tensor(
        [1.0 / (1.0 + abs(s - mean_snr)) for s in client_snrs],
        dtype=torch.float32,
        device=device,
    )  # [n_clients]

    # CHANGE 7: generator quality term from local generator reconstruction loss
    gen_quality = torch.tensor(
        [1.0 / (1.0 + float(L)) for L in client_gen_recon_losses],
        dtype=torch.float32,
        device=device,
    )  # [n_clients]

    # Prototype similarity term (only if prototype banks exist)
    max_P = max((p.size(0) for p in client_prototypes if p is not None and p.numel() > 0), default=0)
    if max_P == 0:
        s_proto = torch.ones(n_clients, dtype=torch.float32, device=device)
        proto_tensor = None
    else:
        proto_dim = client_prototypes[0].size(-1) if client_prototypes[0].numel() > 0 else 1
        proto_tensor = torch.zeros(n_clients, max_P, proto_dim, device=device)
        for i, p in enumerate(client_prototypes):
            if p is None or p.numel() == 0:
                continue
            P = min(p.size(0), max_P)
            proto_tensor[i, :P] = p[:P]

        global_proto_mean = proto_tensor.mean(dim=0)  # [P, D]
        global_flat = F.normalize(global_proto_mean.flatten(), dim=-1)  # [P*D]

        s_proto = torch.zeros(n_clients, dtype=torch.float32, device=device)
        for i in range(n_clients):
            local_flat = F.normalize(proto_tensor[i].flatten(), dim=-1)
            proto_sim = F.cosine_similarity(local_flat, global_flat, dim=0).item()
            s_proto[i] = max(proto_sim, 0.0)

    # Final aggregation weights
    weights = snr_sim * s_proto * gen_quality  # CHANGE 7
    if weights.sum() <= 0:
        weights = torch.ones_like(weights) / len(weights)
    else:
        weights = weights / weights.sum()

    # Aggregate generator parameters
    agg_gen_state = {}
    for key in client_generator_weights[0].keys():
        agg_param = None
        for i in range(n_clients):
            w = weights[i]
            param = client_generator_weights[i][key]
            agg_param = w * param if agg_param is None else (agg_param + w * param)
        agg_gen_state[key] = agg_param

    # Aggregate prototype parameters with the SAME weights (CHANGE 8)
    if proto_tensor is None:
        agg_prototypes = torch.empty(0, device=device)
    else:
        agg_prototypes = (weights.view(n_clients, 1, 1) * proto_tensor).sum(dim=0)  # [P, D]

    return agg_gen_state, agg_prototypes

