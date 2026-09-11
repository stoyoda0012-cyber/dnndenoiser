"""Regression guard: the shipped package carries no developer-specific path.

``AGENTS.md`` §10 forbids developer-specific absolute paths in tracked files,
and the package's modules ship inside the wheel, so a path that leaked into one
of them would be distributed. This file is the guard for that, and it is the
only thing in this file.

The literals below are the guard's test data. They make this file itself a hit
for any scanner looking for private paths, which is why it is allowlisted by
path — the same way ``tests/test_import_boundary.py`` is allowlisted for
sibling-project tokens. Keep the literals as they are: an allowlist that is
keyed to this file's exact content is what stops the allowance from covering
anything else that might be added here later.

Run with:
    pytest tests/test_paths.py -v
"""
from pathlib import Path


def test_library_carries_no_developer_specific_paths():
    """Regression guard: no machine- or developer-specific path in the package.

    ``AGENTS.md`` §10 forbids developer-specific absolute paths in tracked
    files, and these ship inside the wheel.
    """
    import dnndenoiser

    # These literals are test data, so this file is itself a hit for any
    # content scanner looking for private paths — allowlist it by path, the way
    # tests/test_import_boundary.py is allowlisted for sibling-project tokens.
    # The scan covers the installed package, which is what ships; scanning the
    # rest of the tracked tree belongs to the repository-wide boundary scan.
    forbidden = (
        r'C:\Users',
        'C:/Users',
        'Documents/MATLAB',
        'Simulation/Noise2Noise',
        '/Users/',
        '/home/',
    )
    package_root = Path(dnndenoiser.__file__).parent

    offenders = []
    for source in sorted(package_root.rglob('*.py')):
        text = source.read_text(encoding='utf-8')
        for token in forbidden:
            if token in text:
                offenders.append(f'{source.relative_to(package_root)}: {token!r}')

    assert not offenders, 'developer-specific paths in the package: ' + '; '.join(offenders)
