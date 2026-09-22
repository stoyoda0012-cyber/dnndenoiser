#!/usr/bin/env python3
"""
DNNDenoiser Command Line Interface

XPS spectral denoising using deep learning.

Usage:
    dnndenoiser <command> [options]

Commands:
    generate   Generate synthetic XPS spectra
    train      Train denoising model (noise2clean / noise2noise)
    infer      Run inference (denoising)
    evaluate   Evaluate denoising results

Examples:
    # Generate synthetic data
    dnndenoiser generate -o data.h5 -n 1000 --peak-set C1s_adventitious

    # Train model
    dnndenoiser train -d data.h5 -o model.pt --arch FCNN --epochs 30

    # Run inference
    dnndenoiser infer -d noisy.h5 -m model.pt -o denoised.h5

    # Evaluate results
    dnndenoiser evaluate -d denoised.h5 --clean clean.h5
"""

import argparse
import sys
from pathlib import Path


def poisson_level_to_pair_level(level):
    """Convert a ``generate --poisson-level`` value to the Noise2Noise parameter.

    The two are not the same number. ``generate`` draws counts with
    ``lambda = (10000 / level) ** 2`` at the peak, so its relative noise there is
    ``level / 10000``. ``Noise2Noise.generate_pair`` scales counts by
    ``10000 / level`` instead, giving relative noise ``sqrt(level / 10000)``.
    Passing a generate-style level straight through makes the synthesized
    realization noisier than the data it is paired with by
    ``sqrt(10000 / level)`` — tenfold at ``generate``'s default level of 100.

    Equating the two relative noises gives ``level ** 2 / 10000``.

    This assumes each clean spectrum is normalized to its own maximum, which is
    what ``generate`` produces for two-dimensional output: it scales by that
    maximum before drawing counts, while ``generate_pair`` does not. The
    assumption fails under ``--no-normalize``, and for angle- or time-resolved
    output, which normalizes by one global maximum — the array then peaks at 1
    while individual spectra do not, and the conversion is off by
    ``1 / sqrt(peak)`` for each of them.
    """
    return level * level / 10000.0


def cmd_generate(args):
    """Generate synthetic XPS spectra."""
    from dnndenoiser.data.synthetic_generator import (
        SyntheticGenerator, GeneratorConfig, NoiseConfig,
        get_peak_set
    )

    print("=== Synthetic Data Generation ===")
    print(f"Peak set: {args.peak_set}")
    print(f"Samples: {args.n_samples}")
    print(f"Energy points: {args.n_energy}")

    # Build noise config
    noise_config = NoiseConfig(
        noise_type=args.noise_type,
        poisson_level=args.poisson_level,
        gaussian_std=args.gaussian_std,
    )
    print(f"Noise: {noise_config}")

    # Build generator config
    gen_config = GeneratorConfig(
        eta=args.eta,
        use_pseudo_voigt=not args.true_voigt,
        n_energy_points=args.n_energy,
        background_type=args.background,
        background_level=args.bg_level,
        intensity_variation=args.intensity_var,
        position_jitter=args.position_jitter,
        width_variation=args.width_var,
        normalize=not args.no_normalize,
        # Angle-resolved
        n_angles=args.n_angles,
        angle_range=(args.angle_min, args.angle_max),
        angle_intensity_model=args.angle_model,
        # Time-resolved
        n_times=args.n_times,
        time_range=(args.time_min, args.time_max),
        time_intensity_model=args.time_model,
        time_decay_constant=args.time_decay,
    )

    # Determine dimensionality
    if args.n_angles > 1 and args.n_times > 1:
        print("Mode: 4D (samples × times × angles × energy)")
        print(f"  Angles: {args.n_angles} ({args.angle_min}° - {args.angle_max}°)")
        print(f"  Times: {args.n_times} ({args.time_min} - {args.time_max})")
    elif args.n_angles > 1:
        print(f"Mode: 3D angle-resolved ({args.n_angles} angles)")
    elif args.n_times > 1:
        print(f"Mode: 3D time-resolved ({args.n_times} time points)")
    else:
        print("Mode: 2D standard")

    # Get peak set
    peak_set = get_peak_set(args.peak_set)
    print(f"Peaks: {len(peak_set.peaks)}")

    # Generate
    print("\nGenerating...")
    gen = SyntheticGenerator(peak_set, noise_config, gen_config)
    clean, noisy, energy, metadata = gen.generate_batch(args.n_samples, seed=args.seed)

    print(f"  Clean shape: {clean.shape}")
    print(f"  Noisy shape: {noisy.shape}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    h5_path = SyntheticGenerator.save_hdf5(
        output_path, clean, noisy, energy, metadata,
        angles=gen.angles, times=gen.times
    )
    print(f"\nSaved: {h5_path}")

    # Save manifest
    if args.manifest:
        manifest_path = output_path.with_suffix('')
        SyntheticGenerator.save_manifest(manifest_path, metadata, format='jsonl')
        print(f"Manifest: {manifest_path.with_suffix('.jsonl')}")

    print("Done.")


# The two training paths wrote the model's shape under different key names.
# noise2clean/noise2noise used the short spellings; the moving-average path
# introduced the long ones in v0.1.1. Both are published, so the reader accepts
# both -- and reads rather than guesses. The previous code defaulted silently,
# which turned "the checkpoint does not say" into a shape mismatch deep inside
# load_state_dict instead of a sentence naming the file.
_CONFIG_KEYS = {
    'num_features': ('num_features', 'n_features'),
    'num_hidden_units': ('num_hidden_units', 'hidden_units'),
    'encoder_output_dim': ('encoder_output_dim', 'encoder_dim'),
}


def checkpoint_model_config(checkpoint, path):
    """Read the model's shape from a checkpoint, in either key spelling."""
    config = {}
    for canonical, spellings in _CONFIG_KEYS.items():
        for spelling in spellings:
            if spelling in checkpoint:
                config[canonical] = checkpoint[spelling]
                break
        else:
            print(
                f"Error: {path} records no {' or '.join(spellings)}, so the model "
                f"cannot be rebuilt from it. A checkpoint written by "
                f"`dnndenoiser train` always carries this; a file that does not "
                f"was produced some other way.",
                file=sys.stderr,
            )
            sys.exit(1)

    if 'architecture' not in checkpoint:
        print(
            f"Error: {path} records no architecture. Rebuilding the model would "
            f"mean guessing which of the eight it is.",
            file=sys.stderr,
        )
        sys.exit(1)
    config['architecture'] = checkpoint['architecture']
    return config


def load_checkpoint(path, trust=False):
    """Load a checkpoint, refusing to unpickle unless asked to.

    ``torch.load`` defaults to a full unpickle, which **executes code from the
    file**. A denoiser CLI invites exactly the risky case: checkpoints are the
    natural thing to pass around, and ``infer -m someone-elses-model.pt`` would
    run whatever that file says to run.

    So loading is restricted by default, and the unpickler is reached only by
    asking for it. Checkpoints this version writes hold nothing that needs it.
    Those written by v0.1.0 and v0.1.1 stored the energy axis as a NumPy array,
    which the restricted loader rejects — the error names the flag rather than
    leaving the user to find it.
    """
    import torch

    if trust:
        return torch.load(path, map_location='cpu', weights_only=False)
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except Exception as exc:
        print(
            f"Error: {path} cannot be loaded without unpickling it, which runs "
            f"code from the file.\n"
            f"  Underlying cause: {type(exc).__name__}: {str(exc).splitlines()[0]}\n"
            f"  Checkpoints written by dnndenoiser v0.1.2 and later do not need "
            f"this; v0.1.0 and v0.1.1 stored the energy axis as a NumPy array, "
            f"which the restricted loader refuses.\n"
            f"  If you produced this file yourself, or you otherwise trust its "
            f"origin, re-run with --trust-checkpoint. Do not use it on a file "
            f"you were given.",
            file=sys.stderr,
        )
        sys.exit(1)


def _resolve_device(requested):
    """Resolve ``auto`` to the best available backend."""
    import torch

    if requested != 'auto':
        return requested
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


# Flags this method does not take. Its optimiser, schedule, loss and
# architecture are the archived reference's and are part of what is being
# reproduced, so a flag that would change them is refused rather than ignored:
# silently training something else under the method's name is the failure mode
# worth spending an error message on.
#
# Refused on *presence*, not on value. Comparing against the parser default
# would let --arch through in silence, because its default is FCNN while this
# method is always ResNet-FCNN -- so the value a user never touched is already
# the wrong one, and "you did not change it" is not the question. The question
# is whether the flag means anything here, and it does not.
_MOVING_AVERAGE_FIXED = (
    "--arch", "--lr", "--lr-drop-period", "--lr-drop-factor", "--scheduler",
    "--warmup-epochs", "--weight-decay", "--grad-clip", "--hidden-units",
    "--encoder-dim", "--noise-level",
)


def _flags_passed(argv, known_options):
    """Option names actually present in ``argv``, however they were written.

    Three forms all name the same option and a literal membership test sees only
    the first: ``--lr 0.05``, ``--lr=0.05``, and argparse's unique-prefix
    abbreviation ``--weight-deca``. A refusal rule that misses two of the three
    refuses nothing in particular.
    """
    passed = set()
    for token in argv:
        if not token.startswith("-") or token == "--":
            continue
        name = token.split("=", 1)[0]
        if name in known_options:
            passed.add(name)
            continue
        if name.startswith("--"):
            matches = [o for o in known_options if o.startswith(name)]
            if len(matches) == 1:  # argparse accepts a unique prefix
                passed.add(matches[0])
    return passed


def cmd_train_moving_average(args, passed_flags):
    """Self-supervised training from a frame stack, with no clean reference.

    Each frame's target is the mean of its ``--window`` temporally nearest
    *other* frames. Acceptance criteria:
    ``docs/preregistration/P1-selfsupervised-moving-average.md``.
    """
    import numpy as np
    import torch

    from dnndenoiser.data.frame_stack import read_frame_stack
    from dnndenoiser.training.selfsupervised import (
        TARGET_LENGTH,
        moving_average_targets,
        resample,
        train_selfsupervised,
    )

    refused = [flag for flag in _MOVING_AVERAGE_FIXED if flag in passed_flags]
    if refused:
        flags = ', '.join(refused)
        print(
            f"Error: --method moving-average does not take {flags}. Its optimiser, "
            "schedule, loss and architecture are those of the method being "
            "reproduced and are not tunable here; training with different ones "
            "would not be that method. Drop the flag, or use another --method.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.window < 1:
        print(f"Error: --window must be at least 1, got {args.window}", file=sys.stderr)
        sys.exit(1)

    print("\nLoading frame stack...")
    try:
        stack = read_frame_stack(args.data)
    except (KeyError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    contiguous = np.array_equal(stack.frame_index, np.arange(stack.n_frames))
    print(f"  Frames: {stack.n_frames} x {stack.n_energy}")
    print(f"  Acquisition order: {'contiguous' if contiguous else 'explicit'}")

    frames = np.asarray(stack.frames, dtype=np.float32)
    if stack.n_energy != TARGET_LENGTH:
        print(f"  Resampling {stack.n_energy} -> {TARGET_LENGTH} points")
        frames = resample(frames, TARGET_LENGTH)

    # Element-global min-max, as the method's own pipeline does. The constants go
    # into the checkpoint because the model's output lives in the same normalised
    # space and cannot be read back without them.
    g_min, g_max = float(frames.min()), float(frames.max())
    if g_max <= g_min:
        print("Error: the frame stack is constant; there is nothing to normalise "
              "or denoise", file=sys.stderr)
        sys.exit(1)
    normalised = (frames - g_min) / (g_max - g_min)

    window = min(args.window, stack.n_frames - 1)
    if window != args.window:
        print(f"  Window clamped to {window} ({stack.n_frames} frames available)")
    targets = moving_average_targets(normalised, stack.frame_index, window)

    device = _resolve_device(args.device)
    print(f"Device: {device}")
    print(f"\nTraining (window={window}, epochs={args.epochs}, batch={args.batch_size})...")
    model = train_selfsupervised(
        normalised, targets,
        epochs=args.epochs, batch_size=args.batch_size, device=device, seed=args.seed,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": "ResNet-FCNN",
            "num_features": TARGET_LENGTH,
            "num_hidden_units": 100,
            "encoder_output_dim": 64,
            "training_method": "moving-average",
            "window": window,
            "n_frames": stack.n_frames,
            "normalisation": {"min": g_min, "max": g_max, "kind": "element-global min-max"},
            # A tensor, not a NumPy array: an array makes the whole checkpoint
            # unloadable without unpickling it, and nothing else here needs that.
            "energy": torch.as_tensor(np.asarray(stack.energy, dtype=np.float32)),
        },
        args.output,
    )
    print(f"\nSaved: {args.output}")
    print("  Evaluation note: a measured stack has no clean reference. A mean over "
          "the same frames is not independent of these targets, so any SNR computed "
          "against it is not a held-out result.")
    return 0


def cmd_train(args):
    """Train denoising model."""
    if args.method == 'moving-average':
        # A different data layout and fixed hyperparameters: routed to its own
        # command rather than threaded through the generic loop, so the knobs
        # that do not apply cannot silently apply.
        return cmd_train_moving_average(
            args, _flags_passed(sys.argv[1:], args._train_options)
        )

    import torch
    import numpy as np
    import h5py
    from dnndenoiser.models.network import DenoisingNetwork
    from dnndenoiser.training.methods import (
        POISSON_NOISE_FLOOR, TrainingMethodType, create_training_method,
    )

    print("=== Model Training ===")
    print(f"Data: {args.data}")
    print(f"Architecture: {args.arch}")
    print(f"Method: {args.method}")

    # Load data
    print("\nLoading data...")
    with h5py.File(args.data, 'r') as f:
        noisy = f['noisy'][:]
        clean = f['clean'][:] if 'clean' in f else None

    print(f"  Noisy shape: {noisy.shape}")
    if clean is not None:
        print(f"  Clean shape: {clean.shape}")

    # Flatten if multidimensional
    if noisy.ndim > 2:
        n_energy = noisy.shape[-1]
        noisy = noisy.reshape(-1, n_energy)
        if clean is not None:
            clean = clean.reshape(-1, n_energy)
        print(f"  Flattened to: {noisy.shape}")

    n_features = noisy.shape[-1]

    device = _resolve_device(args.device)
    print(f"Device: {device}")

    # Build model directly
    print("\nBuilding model...")
    model = DenoisingNetwork(
        num_features=n_features,
        num_hidden_units=args.hidden_units,
        layer_type=args.arch,
        encoder_output_dim=args.encoder_dim
    )
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Network: {args.arch}")
    print(f"  Parameters: {n_params:,}")

    # Training method
    method_type = TrainingMethodType(args.method)
    training_method = create_training_method(method_type)

    # Check requirements
    if training_method.requires_clean_target and clean is None:
        print(f"Error: {args.method} requires clean data but none found in {args.data}",
              file=sys.stderr)
        sys.exit(1)

    # Prepare data
    print("\nPreparing training data...")
    if method_type == TrainingMethodType.NOISE2CLEAN:
        train_input = torch.tensor(noisy, dtype=torch.float32)
        train_target = torch.tensor(clean, dtype=torch.float32)
    elif method_type == TrainingMethodType.NOISE2NOISE:
        # The CLI synthesizes a second, independent noisy realization from the
        # clean spectra; it does not ingest measured noisy/noisy pairs.
        # The level is stated rather than inferred: the target realization has to
        # sit in the same noise regime as the input, and the data file does not
        # record how it was generated.
        if args.noise_level is None:
            print("Error: --method noise2noise needs --noise-level. The synthesized "
                  "target has to sit in the same noise regime as the input, and the "
                  "data file does not record how it was generated — pass the value "
                  "used for 'generate --poisson-level'.", file=sys.stderr)
            sys.exit(1)
        if not np.isfinite(args.noise_level) or args.noise_level <= 0:
            print(f"Error: --noise-level must be a finite positive number, "
                  f"got {args.noise_level}", file=sys.stderr)
            sys.exit(1)
        pair_level = poisson_level_to_pair_level(args.noise_level)
        if pair_level <= POISSON_NOISE_FLOOR:
            floor = float(np.sqrt(POISSON_NOISE_FLOOR * 10000.0))
            print(f"Error: --noise-level must be greater than {floor:g}; below that no "
                  f"noise is added at all and the target would be the clean spectrum, "
                  f"which is noise2clean training under another name", file=sys.stderr)
            sys.exit(1)
        if not np.isfinite(clean).all():
            print("Error: the clean spectra contain non-finite values, which cannot be "
                  "turned into Poisson rates", file=sys.stderr)
            sys.exit(1)
        # Per spectrum, not over the whole array: angle- and time-resolved generation
        # normalizes by one global maximum, so the array peaks at 1 while individual
        # spectra do not, and a global check would pass exactly when it should not.
        row_peak = np.max(np.abs(clean), axis=1)
        if not np.all((0.95 <= row_peak) & (row_peak <= 1.05)):
            print(f"Warning: individual clean spectra peak between {row_peak.min():.3g} "
                  f"and {row_peak.max():.3g}, not at 1. The synthesized realization is "
                  f"calibrated against spectra normalized to their own maximum, so its "
                  f"noise level is off by 1/sqrt(peak) where they are not. Angle- and "
                  f"time-resolved generation normalizes by a global maximum, and "
                  f"--no-normalize skips normalization entirely.", file=sys.stderr)
        print(f"  Second realization: Poisson level {args.noise_level:g} "
              f"(as passed to generate --poisson-level)")
        noisy2 = np.zeros_like(noisy)
        for i in range(len(clean)):
            pair = training_method.generate_pair(clean[i], pair_level)
            noisy2[i] = pair.target
        train_input = torch.tensor(noisy, dtype=torch.float32)
        train_target = torch.tensor(noisy2, dtype=torch.float32)
    else:
        # noise2self is not offered by the CLI: a correct implementation needs a
        # masked loss that scores only held-out coordinates. The library class
        # (dnndenoiser.training.methods.Noise2Self) is available as experimental code.
        print(f"Error: training method '{args.method}' is not supported by the CLI",
              file=sys.stderr)
        sys.exit(1)

    # DataLoader
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(train_input, train_target)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Optimizer and loss
    if args.weight_decay > 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Scheduler setup
    num_batches = len(dataloader)
    num_training_steps = num_batches * args.epochs

    if args.scheduler == 'cosine':
        from dnndenoiser.training.schedulers import get_cosine_schedule_with_warmup
        num_warmup_steps = num_batches * args.warmup_epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=0.5,
            min_lr_ratio=0.01
        )
        scheduler_step_per_batch = True
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_drop_period, gamma=args.lr_drop_factor
        )
        scheduler_step_per_batch = False

    loss_fn = torch.nn.HuberLoss(delta=1.0)

    # Training loop
    print("\nTraining...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Scheduler: {args.scheduler}")
    if args.scheduler == 'cosine':
        print(f"  Warmup epochs: {args.warmup_epochs}")

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch_input, batch_target in dataloader:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)

            optimizer.zero_grad()
            output = model(batch_input)

            # Handle multi-output models
            if isinstance(output, tuple):
                output = output[0]

            loss = loss_fn(output, batch_target)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            # Step scheduler per batch for cosine
            if scheduler_step_per_batch:
                scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Step scheduler per epoch for StepLR
        if not scheduler_step_per_batch:
            scheduler.step()

        avg_loss = epoch_loss / n_batches
        current_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{args.epochs}: loss = {avg_loss:.6f}, lr = {current_lr:.2e}")

    # Save model
    print("\nSaving model...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'architecture': args.arch,
        'n_features': n_features,
        'hidden_units': args.hidden_units,
        'encoder_dim': args.encoder_dim,
        'training_method': args.method,
        'noise_level': (args.noise_level
                        if method_type == TrainingMethodType.NOISE2NOISE else None),
        'epochs': args.epochs,
        'final_loss': avg_loss,
    }, output_path)

    print(f"Saved: {output_path}")
    print("Done.")


def cmd_infer(args):
    """Run inference (denoising)."""
    import torch
    import numpy as np
    import h5py
    from dnndenoiser.models.network import DenoisingNetwork

    print("=== Inference ===")
    print(f"Data: {args.data}")
    print(f"Model: {args.model}")

    # Load model checkpoint
    print("\nLoading model...")
    checkpoint = load_checkpoint(args.model, trust=args.trust_checkpoint)

    config = checkpoint_model_config(checkpoint, args.model)
    arch = config['architecture']
    n_features = config['num_features']
    hidden_units = config['num_hidden_units']
    encoder_dim = config['encoder_output_dim']

    # The normalisation the model was trained under, if it recorded one. Without
    # applying it the model is fed data on a scale it never saw, and its output
    # is written in a space the file does not name -- wrong twice, silently.
    normalisation = checkpoint.get('normalisation')

    print(f"  Architecture: {arch}")
    print(f"  Features: {n_features}")
    if normalisation is not None:
        print(f"  Normalisation: {normalisation.get('kind', 'unknown')}, "
              f"min={normalisation['min']:.6g} max={normalisation['max']:.6g}")

    device = _resolve_device(args.device)
    print(f"Device: {device}")

    # Build model directly
    model = DenoisingNetwork(
        num_features=n_features,
        num_hidden_units=hidden_units,
        layer_type=arch,
        encoder_output_dim=encoder_dim
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load data
    print("\nLoading data...")
    with h5py.File(args.data, 'r') as f:
        noisy = f['noisy'][:]
        energy = f['energy'][:]
        clean = f['clean'][:] if 'clean' in f else None
        angles = f['angles'][:] if 'angles' in f else None
        times = f['times'][:] if 'times' in f else None

    original_shape = noisy.shape
    print(f"  Input shape: {original_shape}")

    # Flatten if needed
    if noisy.ndim > 2:
        noisy_flat = noisy.reshape(-1, noisy.shape[-1])
    else:
        noisy_flat = noisy

    if normalisation is not None:
        span = normalisation['max'] - normalisation['min']
        if span <= 0:
            print(f"Error: the checkpoint's normalisation is degenerate "
                  f"(min={normalisation['min']:.6g}, max={normalisation['max']:.6g})",
                  file=sys.stderr)
            sys.exit(1)
        noisy_flat = (noisy_flat - normalisation['min']) / span
        print("  Applied the checkpoint's normalisation to the input.")
        print("  The constants are the training stack's, not this file's: if "
              "these spectra sit on a different scale, the model is being fed "
              "data outside the distribution it was trained on.")

    # Inference
    print("\nRunning inference...")
    with torch.no_grad():
        # Keep the full input on host; move one batch at a time so
        # --batch-size bounds device (VRAM) usage for large datasets
        input_tensor = torch.tensor(noisy_flat, dtype=torch.float32)

        batch_size = args.batch_size
        outputs = []

        for i in range(0, len(input_tensor), batch_size):
            batch = input_tensor[i:i+batch_size].to(device)
            output = model(batch)
            if isinstance(output, tuple):
                output = output[0]
            outputs.append(output.cpu().numpy())

        denoised_flat = np.concatenate(outputs, axis=0)

    if normalisation is not None:
        # Back to the input's units. Without this the file would hold numbers in
        # a normalised space it does not name, next to a `noisy` dataset in
        # counts, and nothing would say they are not comparable.
        span = normalisation['max'] - normalisation['min']
        denoised_flat = denoised_flat * span + normalisation['min']
        print("  Inverted the normalisation, so the output is in the input's units.")

    # Reshape back
    denoised = denoised_flat.reshape(original_shape)
    print(f"  Output shape: {denoised.shape}")

    # Save results
    print("\nSaving results...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('denoised', data=denoised, dtype='float32')
        f.create_dataset('noisy', data=noisy.reshape(original_shape), dtype='float32')
        f.create_dataset('energy', data=energy, dtype='float32')
        if clean is not None:
            f.create_dataset('clean', data=clean, dtype='float32')
        if angles is not None:
            f.create_dataset('angles', data=angles, dtype='float32')
        if times is not None:
            f.create_dataset('times', data=times, dtype='float32')

    print(f"Saved: {output_path}")
    print("Done.")


def compute_snr(signal, reference):
    """Truth-referenced SNR in dB: ``10*log10(mean(ref^2) / mean((signal-ref)^2))``.

    The reference is the truth, not an estimate of it, so this needs synthetic
    data or a measured clean spectrum — there is no reference-free SNR for a
    measured one (``AGENTS.md`` §5). Background counts as signal: the statistic
    is over the whole spectrum, not the peak.

    Noise power is floored at ``1e-10`` rather than added to, so the value is
    exact wherever it is meaningful and only the degenerate case is guarded.
    Perfect reconstruction therefore reports a large finite number set by that
    floor, not infinity.
    """
    import numpy as np

    noise = signal - reference
    signal_power = np.mean(reference ** 2, axis=-1)
    noise_power = np.maximum(np.mean(noise ** 2, axis=-1), 1e-10)
    return 10 * np.log10(signal_power / noise_power)


def compute_mse(signal, reference):
    """Mean squared error per sample, over the last axis."""
    import numpy as np

    return np.mean((signal - reference) ** 2, axis=-1)


def cmd_evaluate(args):
    """Evaluate denoising results."""
    import numpy as np
    import h5py

    print("=== Evaluation ===")
    print(f"Data: {args.data}")

    # Load data
    with h5py.File(args.data, 'r') as f:
        denoised = f['denoised'][:] if 'denoised' in f else None
        noisy = f['noisy'][:]
        clean = f['clean'][:] if 'clean' in f else None

    # If clean not in main file, try separate file
    if clean is None and args.clean:
        with h5py.File(args.clean, 'r') as f:
            clean = f['clean'][:]

    if clean is None:
        print("Error: Clean reference data required for evaluation")
        print("  Provide --clean or include 'clean' dataset in input file")
        sys.exit(1)

    if denoised is None:
        print("Error: No 'denoised' dataset found in input file")
        sys.exit(1)

    # Flatten for metrics
    if noisy.ndim > 2:
        noisy = noisy.reshape(-1, noisy.shape[-1])
        denoised = denoised.reshape(-1, denoised.shape[-1])
        clean = clean.reshape(-1, clean.shape[-1])

    print(f"Samples: {len(noisy)}")

    # Input metrics (noisy vs clean)
    snr_input = compute_snr(noisy, clean)
    mse_input = compute_mse(noisy, clean)

    # Output metrics (denoised vs clean)
    snr_output = compute_snr(denoised, clean)
    mse_output = compute_mse(denoised, clean)

    # Gains
    snr_gain = snr_output - snr_input
    mse_reduction = (mse_input - mse_output) / mse_input * 100

    print(f"\n{'Metric':<20} {'Input':<15} {'Output':<15} {'Improvement':<15}")
    print("-" * 65)
    print(f"{'SNR (dB)':<20} {np.mean(snr_input):<15.2f} {np.mean(snr_output):<15.2f} {np.mean(snr_gain):+.2f} dB")
    print(f"{'MSE':<20} {np.mean(mse_input):<15.6f} {np.mean(mse_output):<15.6f} {np.mean(mse_reduction):.1f}% reduction")

    # Per-sample statistics
    print(f"\n{'Statistic':<20} {'SNR Gain (dB)':<15} {'MSE Reduction (%)':<15}")
    print("-" * 50)
    print(f"{'Mean':<20} {np.mean(snr_gain):<15.2f} {np.mean(mse_reduction):<15.1f}")
    print(f"{'Std':<20} {np.std(snr_gain):<15.2f} {np.std(mse_reduction):<15.1f}")
    print(f"{'Min':<20} {np.min(snr_gain):<15.2f} {np.min(mse_reduction):<15.1f}")
    print(f"{'Max':<20} {np.max(snr_gain):<15.2f} {np.max(mse_reduction):<15.1f}")

    # Save metrics if output specified
    if args.output:
        import json
        metrics = {
            'snr_input_mean': float(np.mean(snr_input)),
            'snr_output_mean': float(np.mean(snr_output)),
            'snr_gain_mean': float(np.mean(snr_gain)),
            'snr_gain_std': float(np.std(snr_gain)),
            'mse_input_mean': float(np.mean(mse_input)),
            'mse_output_mean': float(np.mean(mse_output)),
            'mse_reduction_mean': float(np.mean(mse_reduction)),
            'n_samples': len(noisy),
        }

        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved: {output_path}")

    print("\nDone.")


def build_parser():
    """Construct the argument parser.

    Separated from :func:`main` because two things need to inspect it rather
    than only run it: the moving-average refusal rule, which resolves
    abbreviated flags against the registered option strings, and the tests that
    assert what a default actually is.
    """
    parser = argparse.ArgumentParser(
        description='DNNDenoiser - XPS spectral denoising with deep learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate synthetic data
  dnndenoiser generate -o data.h5 -n 1000 --peak-set C1s_adventitious

  # Train model
  dnndenoiser train -d data.h5 -o model.pt --arch FCNN --epochs 30

  # Run inference
  dnndenoiser infer -d noisy.h5 -m model.pt -o denoised.h5

  # Evaluate results
  dnndenoiser evaluate -d denoised.h5
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # === generate ===
    gen_parser = subparsers.add_parser('generate', help='Generate synthetic XPS spectra')
    gen_parser.add_argument('-o', '--output', required=True, help='Output HDF5 file')
    gen_parser.add_argument('-n', '--n-samples', type=int, default=1000, help='Number of samples')
    gen_parser.add_argument('--n-energy', type=int, default=256, help='Energy points')
    gen_parser.add_argument('--peak-set', default='C1s_adventitious',
                           choices=['C1s_single', 'C1s_adventitious', 'C1s_polymer',
                                   'O1s_oxide', 'Si2p_oxide', 'N1s_amine', 'test_doublet'],
                           help='Peak set preset')
    gen_parser.add_argument('--eta', type=float, default=0.3, help='Lorentzian fraction (0-1)')
    gen_parser.add_argument('--true-voigt', action='store_true', help='Use true Voigt (slower)')
    gen_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    # Noise
    gen_parser.add_argument('--noise-type', default='poisson',
                           choices=['none', 'poisson', 'gaussian', 'mixed'])
    gen_parser.add_argument('--poisson-level', type=float, default=100.0,
                           help='Poisson noise level (higher=more noise). Counts '
                                'are drawn from the Poisson distribution; the peak '
                                'bin gets (10000/level)**2 expected counts and every '
                                'other bin a proportional share')
    gen_parser.add_argument('--gaussian-std', type=float, default=0.01,
                           help='Gaussian noise std')
    # Background
    gen_parser.add_argument('--background', default='linear',
                           choices=['none', 'linear', 'shirley'])
    gen_parser.add_argument('--bg-level', type=float, default=0.05)
    # Variation
    gen_parser.add_argument('--intensity-var', type=float, default=0.2)
    gen_parser.add_argument('--position-jitter', type=float, default=0.0)
    gen_parser.add_argument('--width-var', type=float, default=0.1)
    gen_parser.add_argument('--no-normalize', action='store_true')
    gen_parser.add_argument('--manifest', action='store_true', help='Generate manifest file')
    # Angle-resolved
    gen_parser.add_argument('--n-angles', type=int, default=1, help='Number of angles (>1 for 3D)')
    gen_parser.add_argument('--angle-min', type=float, default=0.0)
    gen_parser.add_argument('--angle-max', type=float, default=60.0)
    gen_parser.add_argument('--angle-model', default='cosine',
                           choices=['cosine', 'exponential', 'linear', 'none'])
    # Time-resolved
    gen_parser.add_argument('--n-times', type=int, default=1, help='Number of time points (>1 for 3D)')
    gen_parser.add_argument('--time-min', type=float, default=0.0)
    gen_parser.add_argument('--time-max', type=float, default=100.0)
    gen_parser.add_argument('--time-model', default='exponential_decay',
                           choices=['exponential_decay', 'linear_decay', 'oscillation', 'none'])
    gen_parser.add_argument('--time-decay', type=float, default=50.0)
    gen_parser.set_defaults(func=cmd_generate)

    # === train ===
    train_parser = subparsers.add_parser('train', help='Train denoising model')
    train_parser.add_argument('-d', '--data', required=True, help='Training data HDF5')
    train_parser.add_argument('-o', '--output', required=True, help='Output model file (.pt)')
    train_parser.add_argument('--arch', default='FCNN',
                             choices=['FCNN', 'ResNet-FCNN', '1D-CNN', 'ResNet-1DCNN', 'GRU', 'LSTM', 'bi-LSTM', 'Transformer'],
                             help='Network architecture')
    train_parser.add_argument('--method', default='noise2clean',
                             choices=['noise2clean', 'noise2noise', 'moving-average'],
                             help='Training method. noise2noise synthesizes a second '
                                  'independent noisy realization from the clean spectra. '
                                  '(noise2self is library-only/experimental and not offered here.)')
    train_parser.add_argument('--window', type=int, default=5,
                             help="moving-average only: how many temporally nearest "
                                  "other frames to average into each target. 5 is the "
                                  "canonical value; larger is a longer effective "
                                  "exposure in the target. Clamped to n_frames - 1.")
    train_parser.add_argument('--noise-level', type=float, default=None,
                             help='Required by --method noise2noise: the Poisson level the '
                                  'training data was generated with, in the same units as '
                                  '"generate --poisson-level" (higher means noisier). It sets '
                                  'the noise regime of the synthesized second realization, '
                                  'which has to match the input. There is no default because '
                                  'the data file does not record it. Ignored by noise2clean.')
    train_parser.add_argument('--epochs', type=int, default=30)
    train_parser.add_argument('--batch-size', type=int, default=32)
    train_parser.add_argument('--seed', type=int, default=None,
                             help='Seed passed to torch before the model is built. '
                                  'Construction consumes the random stream, so the '
                                  'same seed gives the same initial weights.')
    train_parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    train_parser.add_argument('--lr-drop-period', type=int, default=10)
    train_parser.add_argument('--lr-drop-factor', type=float, default=0.1)
    train_parser.add_argument('--scheduler', default='step', choices=['step', 'cosine'],
                             help='LR scheduler')
    train_parser.add_argument('--warmup-epochs', type=int, default=5,
                             help='Warmup epochs (for cosine scheduler)')
    train_parser.add_argument('--weight-decay', type=float, default=0.0,
                             help='L2 regularization (AdamW)')
    train_parser.add_argument('--grad-clip', type=float, default=1.0)
    train_parser.add_argument('--hidden-units', type=int, default=100)
    train_parser.add_argument('--encoder-dim', type=int, default=64)
    train_parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    train_parser.set_defaults(
        func=cmd_train,
        # Captured here rather than dug out of the subparser later: the
        # moving-average refusal rule resolves abbreviated flags against the
        # real option set, and this is the one place that set is known.
        _train_options={o for a in train_parser._actions for o in a.option_strings},
    )

    # === infer ===
    infer_parser = subparsers.add_parser('infer', help='Run inference (denoising)')
    infer_parser.add_argument('-d', '--data', required=True, help='Input data HDF5')
    infer_parser.add_argument('-m', '--model', required=True, help='Model file (.pt)')
    infer_parser.add_argument('-o', '--output', required=True, help='Output HDF5')
    infer_parser.add_argument('--batch-size', type=int, default=256)
    infer_parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    infer_parser.add_argument(
        '--trust-checkpoint', action='store_true',
        help='Load the checkpoint with the full unpickler, which RUNS CODE from '
             'the file. Needed for checkpoints written by v0.1.0 and v0.1.1. Use '
             'it only on files you produced or otherwise trust.')
    infer_parser.set_defaults(func=cmd_infer)

    # === evaluate ===
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate denoising results')
    eval_parser.add_argument('-d', '--data', required=True, help='Input HDF5 with denoised data')
    eval_parser.add_argument('--clean', help='Separate clean reference HDF5 (optional)')
    eval_parser.add_argument('-o', '--output', help='Output JSON with metrics')
    eval_parser.set_defaults(func=cmd_evaluate)

    # Parse args
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Run command
    args.func(args)


if __name__ == '__main__':
    main()
