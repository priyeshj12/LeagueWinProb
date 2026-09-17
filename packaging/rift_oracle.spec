# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for rift_oracle.exe.

Produces a single self-contained executable with the trained model bundled
inside it, so the binary can be dropped anywhere and run with nothing beside
it but an API key.

    pyinstaller packaging/rift_oracle.spec

Everything the tool needs at runtime is pure Python plus numpy, requests and
rich. The excludes below drop the scientific-stack extras PyInstaller would
otherwise pull in through numpy and which this tool never imports; they save
roughly half the binary size.
"""

from pathlib import Path

# __file__ is not defined while PyInstaller execs a spec, so derive the project
# root from the spec path it does provide.
PROJECT_ROOT = Path(SPECPATH).resolve().parent

datas = [
    (str(PROJECT_ROOT / "rift_oracle" / "data"), "rift_oracle/data"),
]

a = Analysis(
    [str(PROJECT_ROOT / "rift_oracle" / "__main__.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        # numpy's random bit generators are compiled Cython, so PyInstaller's
        # static scan cannot see the plain `import secrets` inside them. It
        # normally arrives as a side effect of cryptography being bundled;
        # excluding cryptography takes it away, and every simulated game then
        # dies on ModuleNotFoundError. Ask for it directly instead.
        "secrets",
        "rift_oracle.cli",
        "rift_oracle.analysis.advice",
        "rift_oracle.analysis.narrate",
        "rift_oracle.analysis.swings",
        "rift_oracle.game.live_adapter",
        "rift_oracle.game.timeline_adapter",
        "rift_oracle.model.train",
        "rift_oracle.sim.synth",
        "rift_oracle.ui.dashboard",
        "rift_oracle.ui.html_report",
        "rift_oracle.ui.report",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Scientific-stack extras PyInstaller reaches for through numpy and
        # that this tool never imports. Dropping them roughly halves the size.
        "matplotlib", "scipy", "pandas", "sklearn", "PIL", "tkinter",
        "IPython", "jupyter", "pytest", "setuptools", "pydoc_data",
        "numpy.distutils", "numpy.f2py", "numpy.testing",
        # urllib3 v2 does TLS through the standard library's ssl module, so
        # the legacy pyOpenSSL path and its cryptography dependency are dead
        # weight - tens of megabytes of it. Excluding them also sidesteps
        # PyInstaller's module scan importing cryptography's Rust extension in
        # an isolated subprocess, which panics on some Linux builds.
        "cryptography", "OpenSSL",
        "urllib3.contrib.pyopenssl", "urllib3.contrib.securetransport",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="rift_oracle",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
