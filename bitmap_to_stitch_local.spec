# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


root = Path.cwd()


a = Analysis(
    ["local_server.py"],
    pathex=[str(root)],
    binaries=[],
    datas=[("web", "web")],
    hiddenimports=["numpy", "PIL", "PIL.Image", "PIL._tkinter_finder"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BitmapToStitch",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="BitmapToStitch",
)
