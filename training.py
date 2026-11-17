import csv
import json
import math
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

from evaluation import Evaluator, image_tensors, load_inception
from model import Discriminator, Generator


PAUSE_REQUEST = "pause.request"
STEP_COLUMNS = ("epoch", "step", "images", "seconds", "images_per_second", "d_loss", "g_loss")
EPOCH_COLUMNS = STEP_COLUMNS + (
    "peak_allocated_gib", "peak_reserved_gib", "fid", "inception_score",
    "validation_d_loss", "validation_g_loss", "validation_d_accuracy",
    "validation_real_accuracy", "validation_fake_accuracy", "real_images", "generated_images",
    "evaluation_seconds", "is_best",
)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 2
    train_count: int = 75_000
    validation_count: int = 5_000
    batch_size: int = 64
    num_workers: int = 8
    prefetch_factor: int | None = None
    latent_dim: int = 100
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.5, 0.999)
    seed: int = 285
    log_every: int = 50
    eval_every: int = 5
    eval_batch_size: int = 64


class PaintingDataset(Dataset):
    def __init__(self, paths, image_size=256):
        self.paths = paths
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            # Decode at a reduced JPEG resolution when the short edge allows it.
            width, height = image.size
            scale = self.image_size / min(width, height)
            image.draft("RGB", (int(width * scale), int(height * scale)))
            rgb = ImageOps.fit(image.convert("RGB"), (self.image_size, self.image_size), Image.Resampling.LANCZOS)
            pixels = torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8)
        return pixels.view(self.image_size, self.image_size, 3).permute(2, 0, 1)


def make_split(data_dir, config):
    paths = sorted(path for path in data_dir.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg"})
    if len(paths) < config.train_count + config.validation_count:
        raise ValueError(f"Only {len(paths):,} JPEGs found in {data_dir}")
    random.Random(config.seed).shuffle(paths)
    val_end = config.train_count + config.validation_count
    return paths[:config.train_count], paths[config.train_count:val_end]


def make_loader(paths, config, device, *, validation=False):
    dataset = PaintingDataset(paths)
    batch_size = config.eval_batch_size if validation else config.batch_size
    workers = min(4, config.num_workers) if validation else config.num_workers
    sampler = None if validation else RandomSampler(dataset, generator=torch.Generator().manual_seed(config.seed))
    prefetch = None
    if workers > 0 and not validation:
        prefetch = config.prefetch_factor if config.prefetch_factor is not None else (1 if batch_size >= 256 else 2)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        generator=torch.Generator().manual_seed(config.seed + 1),
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=prefetch,
    )


@torch.inference_mode()
def save_samples(generator, noise, path, use_bf16):
    generator.eval()
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            images = generator(noise)
        images = images.float().add(1).mul(127.5).round().clamp(0, 255).to(torch.uint8)
    finally:
        generator.train()
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    size = images.shape[1]
    grid = Image.new("RGB", (6 * size, 6 * size))
    for index, image in enumerate(images):
        grid.paste(Image.fromarray(image), (index % 6 * size, index // 6 * size))
    atomic_write(path, lambda tmp: grid.save(tmp, format="PNG"))


def make_models_and_optimizers(config, device):
    generator = Generator(config.latent_dim).to(device=device, memory_format=torch.channels_last)
    discriminator = Discriminator().to(device=device, memory_format=torch.channels_last)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=config.learning_rate, betas=config.betas, fused=True)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config.learning_rate, betas=config.betas, fused=True)
    return generator, discriminator, optimizer_g, optimizer_d


def train_step(batch, generator, discriminator, optimizer_g, optimizer_d, latent_dim, use_bf16):
    device = next(generator.parameters()).device
    real = image_tensors(batch, device)
    noise = torch.randn(real.size(0), latent_dim, 1, 1, device=device)

    optimizer_d.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        fake = generator(noise)
        real_scores = discriminator(real)
        fake_scores = discriminator(fake.detach())
        d_loss = F.softplus(-real_scores.float()).mean() + F.softplus(fake_scores.float()).mean()
    d_loss.backward()
    optimizer_d.step()

    discriminator.requires_grad_(False)
    optimizer_g.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        g_loss = F.softplus(-discriminator(fake).float()).mean()
    g_loss.backward()
    optimizer_g.step()
    discriminator.requires_grad_(True)
    return torch.stack((d_loss.detach(), g_loss.detach()))


def atomic_write(path, writer):
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        writer(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_write(path, lambda temporary: temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False), encoding="utf-8",
    ))


def write_history(run_dir, history):
    def write_csv(temporary):
        with temporary.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=EPOCH_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows({**row, **row.get("evaluation", {})} for row in history)

    write_json(run_dir / "timing.json", history)
    atomic_write(run_dir / "epochs.csv", write_csv)


def report_epoch(run_dir, history, models, evaluator, use_bf16):
    row = history[-1]
    epoch = row["epoch"]
    snapshot = run_dir / f"epoch_{epoch:04}.pt"
    report_path = run_dir / f"evaluation_{epoch:04}.json"
    if report_path.exists():
        row["evaluation"] = json.loads(report_path.read_text(encoding="utf-8"))
        return

    print(f"Evaluating epoch {epoch} on the held-out split…", flush=True)
    generator, discriminator, optimizer_g, optimizer_d = models
    optimizer_g.zero_grad(set_to_none=True)
    optimizer_d.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    metrics = evaluator.run(generator, discriminator, use_bf16)
    torch.cuda.empty_cache()

    previous_fids = (past["evaluation"]["fid"] for past in history[:-1] if "evaluation" in past)
    is_best = metrics["fid"] < min(previous_fids, default=math.inf)
    if is_best:
        atomic_write(run_dir / "best.pt", lambda temporary: shutil.copyfile(snapshot, temporary))
    row["evaluation"] = {"epoch": epoch, **metrics, "is_best": is_best}
    write_json(report_path, row["evaluation"])
    print(f"Epoch {epoch} | FID={metrics['fid']:.3f} | IS={metrics['inception_score']:.3f} | "
          f"val D loss={metrics['validation_d_loss']:.3f} | val G loss={metrics['validation_g_loss']:.3f} | "
          f"val D acc={metrics['validation_d_accuracy']:.1%} "
          f"(real={metrics['validation_real_accuracy']:.1%}, fake={metrics['validation_fake_accuracy']:.1%}) | "
          f"{metrics['evaluation_seconds']:.1f}s eval" + (" | new best.pt" if is_best else ""), flush=True)


def finish_epoch(run_dir, history, models, evaluator, fixed_noise, use_bf16, eval_every):
    epoch = history[-1]["epoch"]
    evaluate = epoch % eval_every == 0
    snapshot = run_dir / f"epoch_{epoch:04}.pt"
    if evaluate and not snapshot.exists():
        atomic_write(snapshot, lambda temporary: shutil.copyfile(run_dir / "latest.pt", temporary))

    preview = run_dir / f"samples_{epoch:04}.png"
    if not preview.exists():
        save_samples(models[0], fixed_noise, preview, use_bf16)
    write_history(run_dir, history)
    print(f"Epoch {epoch} sample grid: {preview}", flush=True)

    if evaluate:
        report_epoch(run_dir, history, models, evaluator, use_bf16)
        write_history(run_dir, history)


def restore_checkpoint(checkpoint, models, loader, config):
    saved = checkpoint["config"]
    configurable = {"epochs", "num_workers", "prefetch_factor", "log_every"}
    changes = [key for key, value in asdict(config).items() if key not in configurable and saved[key] != value]
    if changes:
        raise ValueError(f"Resume requires the saved values for: {', '.join(changes)}")
    if config.epochs < checkpoint["epoch"]:
        raise ValueError("epochs must be at least the checkpoint's completed epoch")
    for name, component in zip(("generator", "discriminator", "optimizer_g", "optimizer_d"), models):
        component.load_state_dict(checkpoint[name])
    torch.set_rng_state(checkpoint["torch_rng"])
    torch.cuda.set_rng_state(checkpoint["cuda_rng"])
    loader.sampler.generator.set_state(checkpoint["sampler_rng"])


def discard_unfinished_log_rows(path, completed):
    if not path.exists():
        return
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames
        rows = [row for row in reader if int(row["epoch"]) <= completed]

    def write(temporary):
        with temporary.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    atomic_write(path, write)


def request_pause(run_dir):
    """Ask a running training session to pause after its next completed epoch."""
    run_dir = Path(run_dir).resolve()
    if not (run_dir / "config.json").is_file():
        raise FileNotFoundError(f"No initialized training run at {run_dir}")
    request = run_dir / PAUSE_REQUEST
    request.touch()
    return request


def train(data_dir, run_dir, config=TrainConfig(), *, resume=False):
    """Train in a new run folder or continue from that folder's latest.pt."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the 256×256 timing run")

    device = torch.device("cuda")
    use_bf16 = torch.cuda.is_bf16_supported()
    if config.eval_every < 1 or config.validation_count < 2:
        raise ValueError("eval_every must be positive and FID needs at least two held-out images")
    data_dir, run_dir = Path(data_dir).resolve(), Path(run_dir).resolve()
    train_paths, validation_paths = make_split(data_dir, config)
    split = {key: [path.relative_to(data_dir).as_posix() for path in paths]
             for key, paths in {"train": train_paths, "validation": validation_paths}.items()}
    if resume:
        saved_split = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
        if split != saved_split:
            raise ValueError("Current train/validation split differs from this run's saved split")
        checkpoint = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=True)
    elif run_dir.exists():
        raise FileExistsError(f"{run_dir} already exists. Choose a new directory or pass resume=True.")

    evaluator = None
    if config.epochs >= config.eval_every:
        cache_dir = Path(__file__).resolve().parent / ".cache" / "inception"
        validation_loader = make_loader(validation_paths, config, device, validation=True)
        evaluator = Evaluator(load_inception(cache_dir), validation_loader, config.latent_dim, config.seed + 2)
    torch.manual_seed(config.seed)
    loader = make_loader(train_paths, config, device)
    models = make_models_and_optimizers(config, device)
    generator, discriminator, optimizer_g, optimizer_d = models
    preview_rng = torch.Generator(device=device).manual_seed(config.seed + 3)
    fixed_noise = torch.randn(36, config.latent_dim, 1, 1, device=device, generator=preview_rng)
    steps_path = run_dir / "steps.csv"
    pause_request = run_dir / PAUSE_REQUEST
    completed, history = 0, []
    if resume:
        restore_checkpoint(checkpoint, models, loader, config)
        completed, history = checkpoint["epoch"], checkpoint["history"]
        del checkpoint
        discard_unfinished_log_rows(steps_path, completed)
        pause_request.unlink(missing_ok=True)
        finish_epoch(run_dir, history, models, evaluator, fixed_noise, use_bf16, config.eval_every)
        print(f"Resuming from epoch {completed} in {run_dir}", flush=True)
    else:
        run_dir.mkdir(parents=True)
        write_json(run_dir / "split.json", split)
    write_json(run_dir / "config.json", asdict(config))
    print(f"{torch.cuda.get_device_name()}; precision={'BF16 autocast' if use_bf16 else 'FP32'}", flush=True)
    print(f"{len(train_paths):,} train / {len(validation_paths):,} held out; "
          f"{len(loader):,} steps/epoch; batch={config.batch_size}; "
          f"workers={loader.num_workers}; pin_memory={loader.pin_memory}; "
          f"persistent_workers={loader.persistent_workers}; prefetch_factor={loader.prefetch_factor}", flush=True)

    write_header = not steps_path.exists()
    with steps_path.open("a", encoding="utf-8", newline="") as log:
        writer = csv.DictWriter(log, fieldnames=STEP_COLUMNS)
        if write_header:
            writer.writeheader()
        for epoch in range(completed + 1, config.epochs + 1):
            if history and pause_request.is_file():
                print(f"Paused after epoch {history[-1]['epoch']}. Resume from {run_dir / 'latest.pt'}.", flush=True)
                break
            torch.cuda.reset_peak_memory_stats()
            start = perf_counter()
            totals = torch.zeros(2, device=device)
            count = 0
            for step, batch in enumerate(loader, 1):
                if epoch == 1 and step == 1:
                    print(f"First batch loaded in {perf_counter() - start:.1f}s; pinned={batch.is_pinned()}", flush=True)
                batch_size = batch.size(0)
                losses = train_step(batch, *models, config.latent_dim, use_bf16)
                totals.add_(losses, alpha=batch_size)
                count += batch_size
                if step == 1 or step % config.log_every == 0 or step == len(loader):
                    d_mean, g_mean = (totals / count).tolist()
                    if not (math.isfinite(d_mean) and math.isfinite(g_mean)):
                        raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}, step {step}")
                    elapsed = perf_counter() - start
                    row = dict(epoch=epoch, step=step, images=count, seconds=elapsed,
                               images_per_second=count / elapsed, d_loss=d_mean, g_loss=g_mean)
                    writer.writerow(row)
                    log.flush()
                    print(f"Epoch {epoch}/{config.epochs} step {step}/{len(loader)} | {elapsed:.1f}s | "
                          f"{count / elapsed:.1f} img/s | D={d_mean:.3f} G={g_mean:.3f}", flush=True)

            peak_gib = torch.cuda.max_memory_allocated() / 2**30
            reserved_gib = torch.cuda.max_memory_reserved() / 2**30
            history.append({**row, "peak_allocated_gib": peak_gib, "peak_reserved_gib": reserved_gib})
            checkpoint = {
                "epoch": epoch,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
                "sampler_rng": loader.sampler.generator.get_state(),
                "config": asdict(config),
                "history": history,
            }
            atomic_write(run_dir / "latest.pt", lambda temporary: torch.save(checkpoint, temporary))
            print(f"Epoch {epoch} complete: {elapsed:.1f}s; peak GPU memory "
                  f"{peak_gib:.2f} GiB allocated / {reserved_gib:.2f} GiB reserved", flush=True)
            finish_epoch(run_dir, history, models, evaluator, fixed_noise, use_bf16, config.eval_every)
    return history
