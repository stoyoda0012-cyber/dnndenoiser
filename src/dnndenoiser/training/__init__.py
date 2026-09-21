"""Training methods and schedules for XPS spectral denoising.

Import the pieces from their own modules:
- ``dnndenoiser.training.methods``: noise2clean / noise2noise training methods
- ``dnndenoiser.training.schedulers``: learning-rate schedules
- ``dnndenoiser.training.selfsupervised``: leave-one-out moving-average targets
  from a stack of repeated acquisitions, for training without a clean reference
"""
