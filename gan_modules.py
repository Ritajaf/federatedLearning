import copy
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import KMeans


class SemanticGenerator(nn.Module):
    """
    Conditional generator that maps a semantic embedding to a
    channel-robust embedding conditioned on channel state and prototype.

    Inputs:
        semantic_emb:  [batch, semantic_dim]
        channel_state: [batch, channel_dim]  (e.g. SNR as a 1D feature)
        prototype_vec: [batch, prototype_dim]
    Output:
        [batch, output_dim] (usually same as semantic_dim)
    """

    def __init__(self, semantic_dim: int, channel_dim: int, prototype_dim: int, output_dim: int):
        super().__init__()
        in_dim = semantic_dim + channel_dim + prototype_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim),
        )

    def forward(self, semantic_emb: torch.Tensor, channel_state: torch.Tensor, prototype_vec: torch.Tensor) -> torch.Tensor:
        x = torch.cat([semantic_emb, channel_state, prototype_vec], dim=-1)
        return self.net(x)


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
        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
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
    global_prototypes: torch.Tensor,
    pad_idx: int,
    snr: float,
    n_disc_steps: int = 5,
    gen_lr: float = 1e-4,
    disc_lr: float = 1e-4,
) -> Tuple[dict, float, torch.Tensor]:
    """
    Two-timescale local training for conditional GAN on top of the DeepSC encoder.

    This does NOT change the main DeepSC training loop; it only trains the
    generator/discriminator in semantic space and returns:
        - updated generator state_dict (for federated aggregation)
        - client_snr (the scalar SNR used during GAN training)
        - local prototype bank built from semantic embeddings
    """
    device = next(client_model.parameters()).device
    client_model.eval()  # encoder is used as a fixed feature extractor here
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=gen_lr)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=disc_lr)

    collected_embeddings: List[torch.Tensor] = []

    channel_state_value = float(snr)

    for batch in dataloader:
        src = batch.to(device)

        # sentence-level embeddings from encoder
        with torch.no_grad():
            src_mask = (src == pad_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)
            enc_output = client_model.encoder(src, src_mask)  # [B, T, d_model]
            semantic_emb = enc_output.mean(dim=1)             # [B, d_model]
        collected_embeddings.append(semantic_emb.detach())

        # prepare conditional inputs
        B, d_model = semantic_emb.shape
        channel_state = torch.full((B, 1), channel_state_value, device=device)
        prototype_vec = get_nearest_prototype(semantic_emb, global_prototypes)

        # -------- Train discriminator (inner loop) --------
        for _ in range(n_disc_steps):
            fake_emb = generator(semantic_emb, channel_state, prototype_vec)
            real_score = discriminator(semantic_emb, semantic_emb)
            fake_score = discriminator(semantic_emb, fake_emb.detach())

            eps = 1e-8
            disc_loss = -torch.log(real_score + eps) - torch.log(1.0 - fake_score + eps)
            disc_loss = disc_loss.mean()

            disc_optimizer.zero_grad()
            disc_loss.backward()
            disc_optimizer.step()

        # -------- Train generator (outer step) --------
        fake_emb = generator(semantic_emb, channel_state, prototype_vec)
        fake_score = discriminator(semantic_emb, fake_emb)

        if global_prototypes is not None and global_prototypes.numel() > 0:
            global_proto = global_prototypes.mean(dim=0, keepdim=True).expand_as(prototype_vec)
        else:
            global_proto = torch.zeros_like(prototype_vec)

        gen_loss = client_loss(
            original=semantic_emb,
            reconstructed=fake_emb,
            disc_output=fake_score,
            global_prototype=global_proto,
            local_prototype=prototype_vec,
        )

        gen_optimizer.zero_grad()
        gen_loss.backward()
        gen_optimizer.step()

    if collected_embeddings:
        all_emb = torch.cat(collected_embeddings, dim=0)
        local_prototypes = build_prototype_bank(all_emb)
    else:
        local_prototypes = torch.empty(0, device=device)

    return copy.deepcopy(generator.state_dict()), channel_state_value, local_prototypes


def federated_aggregate_generator(
    client_generator_weights: List[dict],
    client_snrs: List[float],
    client_prototypes: List[torch.Tensor],
) -> dict:
    """
    Federated aggregation of generator parameters only, using
    channel- and semantic-similarity based weights.

    This follows the high-level idea from the user description:
        - Compute similarity between each client and the global average
          using SNR distance and cosine similarity of prototype banks.
        - Use these similarities to weight each client's generator parameters.
    """
    n_clients = len(client_generator_weights)
    if n_clients == 0:
        raise ValueError("client_generator_weights is empty.")

    device = next(iter(client_generator_weights[0].values())).device

    # mean SNR across clients
    mean_snr = float(sum(client_snrs) / len(client_snrs))

    # stack prototype banks into a single tensor of shape [n_clients, P, D]
    # if some clients have empty prototype banks, we treat them as zeros
    max_P = max((p.size(0) for p in client_prototypes if p is not None and p.numel() > 0), default=0)
    if max_P == 0:
        # fall back to SNR-only weighting
        similarity_weights = torch.tensor(
            [1.0 / (1.0 + abs(s - mean_snr)) for s in client_snrs],
            dtype=torch.float32,
            device=device,
        )
    else:
        proto_dim = client_prototypes[0].size(-1) if client_prototypes[0].numel() > 0 else 1
        proto_tensor = torch.zeros(n_clients, max_P, proto_dim, device=device)
        for i, p in enumerate(client_prototypes):
            if p is None or p.numel() == 0:
                continue
            P = min(p.size(0), max_P)
            proto_tensor[i, :P] = p[:P]

        global_proto_mean = proto_tensor.mean(dim=0, keepdim=True)  # [1, P, D]
        global_flat = F.normalize(global_proto_mean.flatten(1), dim=-1)  # [1, P*D]

        similarity_weights = torch.zeros(n_clients, dtype=torch.float32, device=device)

        for i in range(n_clients):
            # SNR similarity (closer to mean_snr is better)
            snr_sim = 1.0 / (1.0 + abs(client_snrs[i] - mean_snr))

            # Prototype similarity
            local_flat = F.normalize(proto_tensor[i].flatten().unsqueeze(0), dim=-1)  # [1, P*D]
            proto_sim = F.cosine_similarity(local_flat, global_flat).item()

            similarity_weights[i] = snr_sim * max(proto_sim, 0.0)

    # normalize weights
    if similarity_weights.sum() <= 0:
        # degenerate case, fall back to uniform
        similarity_weights = torch.ones_like(similarity_weights) / len(similarity_weights)
    else:
        similarity_weights = similarity_weights / similarity_weights.sum()

    # weighted aggregation of generator parameters ONLY
    aggregated = {}
    for key in client_generator_weights[0].keys():
        agg_param = None
        for i in range(n_clients):
            w = similarity_weights[i]
            param = client_generator_weights[i][key]
            if agg_param is None:
                agg_param = w * param
            else:
                agg_param = agg_param + w * param
        aggregated[key] = agg_param

    return aggregated

