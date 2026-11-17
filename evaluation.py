import math
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F


def load_inception(cache_dir):
    from torchvision.models import Inception_V3_Weights, inception_v3

    weights = Inception_V3_Weights.IMAGENET1K_V1
    state = torch.hub.load_state_dict_from_url(
        weights.url, model_dir=str(cache_dir), map_location="cpu", check_hash=True, weights_only=True,
    )
    network = inception_v3(weights=None, aux_logits=True, transform_input=True, init_weights=False)
    network.load_state_dict(state)
    network.AuxLogits = None
    classifier = network.fc
    network.fc = nn.Identity()
    return network.eval().requires_grad_(False), classifier.eval().requires_grad_(False)


def image_tensors(batch, device):
    return batch.to(device, non_blocking=True, memory_format=torch.channels_last).float().mul_(2 / 255).sub_(1)


def feature_stats(batches):
    features = torch.cat(batches).double()
    mean = features.mean(0)
    centered = features - mean
    return mean, centered.T.mm(centered) / (len(centered) - 1)


def covariance_root(covariance):
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    return (eigenvectors * eigenvalues.clamp_min(0).sqrt()) @ eigenvectors.T


def frechet_distance(real_mean, real_covariance, real_root, fake_mean, fake_covariance):
    product = real_root @ fake_covariance @ real_root
    eigenvalues = torch.linalg.eigvalsh((product + product.T) * 0.5)
    score = (real_mean - fake_mean).square().sum() + real_covariance.trace() + fake_covariance.trace()
    return (score - 2 * eigenvalues.clamp_min(0).sqrt().sum()).clamp_min(0).item()


class Evaluator:
    def __init__(self, inception, validation_loader, latent_dim, seed):
        self.network, self.classifier = inception
        self.loader = validation_loader
        self.latent_dim = latent_dim
        self.seed = seed
        self.real_stats = None

    @torch.inference_mode()
    def run(self, generator, discriminator, use_bf16):
        from ignite.metrics import InceptionScore

        start = perf_counter()
        device = next(generator.parameters()).device
        rng = torch.Generator(device=device).manual_seed(self.seed)
        network = self.network.to(device=device, memory_format=torch.channels_last)
        classifier = self.classifier.to(device)
        mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)

        def features(images):
            unit = images.float().add(1).mul_(0.5).clamp_(0, 1)
            resized = F.interpolate(unit, size=(299, 299), mode="bilinear", align_corners=False, antialias=True)
            return network((resized - mean) / std)

        real_features, fake_features = [], []
        inception_score = InceptionScore(num_features=1000)
        totals = torch.zeros(5, device=device, dtype=torch.float64)
        count = 0
        generator.eval()
        discriminator.eval()
        try:
            for real_batch in self.loader:
                real = image_tensors(real_batch, device)
                size = real.size(0)
                noise = torch.randn(size, self.latent_dim, 1, 1, device=device, generator=rng)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    fake = generator(noise)
                    d_real = discriminator(real).float()
                    d_fake = discriminator(fake).float()

                totals += torch.stack((
                    F.softplus(-d_real).sum(), F.softplus(d_fake).sum(), F.softplus(-d_fake).sum(),
                    (d_real >= 0).sum(), (d_fake < 0).sum(),
                ))
                if self.real_stats is None:
                    real_features.append(features(real).cpu())
                fake_pool = features(fake)
                fake_features.append(fake_pool.cpu())
                inception_score.update(classifier(fake_pool).softmax(1))
                count += size

            if self.real_stats is None:
                real_mean, real_cov = feature_stats(real_features)
                self.real_stats = real_mean, real_cov, covariance_root(real_cov)
            real_mean, real_cov, real_root = self.real_stats
            fake_mean, fake_cov = feature_stats(fake_features)
            fid = frechet_distance(real_mean, real_cov, real_root, fake_mean, fake_cov)
            real_loss, fake_loss, generator_loss, real_accuracy, fake_accuracy = (totals / count).tolist()
            metrics = {
                "fid": fid,
                "inception_score": inception_score.compute(),
                "validation_d_loss": real_loss + fake_loss,
                "validation_g_loss": generator_loss,
                "validation_d_accuracy": (real_accuracy + fake_accuracy) / 2,
                "validation_real_accuracy": real_accuracy,
                "validation_fake_accuracy": fake_accuracy,
            }
            if not all(math.isfinite(value) for value in metrics.values()):
                raise FloatingPointError(f"Nonfinite validation metrics: {metrics}")
            return {**metrics, "real_images": count, "generated_images": count,
                    "evaluation_seconds": perf_counter() - start}
        finally:
            generator.train()
            discriminator.train()
            network.cpu()
            classifier.cpu()
