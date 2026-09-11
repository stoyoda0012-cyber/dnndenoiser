## Summary

Describe what changed and why.

## Validation

- [ ] Added or updated tests appropriate to the change.
- [ ] Ran `pytest`.
- [ ] Ran `ruff check src/ tests/`.
- [ ] Updated `[Unreleased]` in `CHANGELOG.md`, or explained why no entry is needed.

## Boundaries and scientific claims

- [ ] The change adds no sibling-project imports, private paths, measured-data
      assets, or instrument-specific constants to the library.
- [ ] The change does not present denoised output as a measurement or claim
      reference-free SNR for measured spectra.
- [ ] Any changed scientific model, metric meaning, published claim, or
      public/private distribution boundary received the required independent
      audit.

## Notes

List linked issues, compatibility implications, and any intentionally deferred
work.
