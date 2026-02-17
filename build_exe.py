"""
Build EXE for PSA Gray Zone Prediction Desktop App.

Usage:
    python build_exe.py

Requirements:
    pip install pyinstaller

Output:
    dist/PSA_Predictor/ directory containing the .exe and all dependencies
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent


def main():
    print("=" * 50)
    print("Building PSA Predictor EXE")
    print("=" * 50)

    # Check pyinstaller
    try:
        import PyInstaller
        print(f"PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "PSA_Predictor",
        "--onedir",
        "--windowed",
        "--noconfirm",
        "--clean",
        "--add-data", f"{BASE_DIR / 'model'}:model",
        "--hidden-import", "sklearn.utils._typedefs",
        "--hidden-import", "sklearn.utils._heap",
        "--hidden-import", "sklearn.utils._sorting",
        "--hidden-import", "sklearn.utils._vector_sentinel",
        "--hidden-import", "sklearn.neighbors._partition_nodes",
        "--hidden-import", "sklearn.tree._utils",
        "--hidden-import", "sklearn.ensemble._gradient_boosting",
        "--hidden-import", "sklearn.linear_model._logistic",
        "--collect-submodules", "sklearn",
        str(BASE_DIR / "desktop_app.py"),
    ]

    print(f"\nRunning: {' '.join(cmd[:6])}...")
    subprocess.run(cmd, check=True)

    dist_dir = BASE_DIR / "dist" / "PSA_Predictor"
    if dist_dir.exists():
        print(f"\nBuild successful!")
        print(f"EXE location: {dist_dir}")
        print(f"Run: {dist_dir / 'PSA_Predictor.exe'}")
    else:
        print("\nBuild may have failed. Check output above.")


if __name__ == "__main__":
    main()
