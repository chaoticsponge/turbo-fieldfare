"""Compile the pinned MLX shaders beside the SwiftPM executables.

Xcode handles .metal compilation automatically; command-line SwiftPM does not.
MLX searches for mlx.metallib beside the executable before other locations.
"""
from pathlib import Path
import hashlib
import subprocess


def main():
    root = Path(__file__).resolve().parent.parent
    shaders = root / ".build/checkouts/mlx-swift/Source/Cmlx/mlx-generated/metal"
    if not shaders.is_dir():
        raise SystemExit("Resolve the pinned MLX Swift dependencies first.")
    output = Path(subprocess.check_output(
        ["swift", "build", "-c", "release", "--show-bin-path"], cwd=root, text=True).strip())
    intermediates = root / ".build/qwen-metal"
    intermediates.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256()
    signature.update(subprocess.check_output(["xcrun", "metal", "--version"]))
    sources = sorted(shaders.rglob("*.metal"))
    for path in sorted(shaders.rglob("*")):
        if path.is_file():
            signature.update(str(path.relative_to(shaders)).encode())
            signature.update(path.read_bytes())
    signature.update(b"metal3.2-O2-v1")
    stamp = intermediates / "signature.txt"
    library = output / "mlx.metallib"
    digest = signature.hexdigest()
    if library.is_file() and stamp.is_file() and stamp.read_text() == digest:
        print("MLX Metal library is up to date.")
        return
    objects = []
    for index, source in enumerate(sources):
        target = intermediates / f"{index}.air"
        print(f"Compiling MLX shader {index + 1}/{len(sources)}: {source.name}", flush=True)
        subprocess.run(["xcrun", "-sdk", "macosx", "metal", "-std=metal3.2",
                        "-O2", "-I", str(shaders), "-c", str(source), "-o", str(target)], check=True)
        objects.append(str(target))
    temporary = intermediates / "mlx.metallib"
    subprocess.run(["xcrun", "-sdk", "macosx", "metallib", *objects, "-o", str(temporary)], check=True)
    temporary.replace(library)
    stamp.write_text(digest)
    print(f"Built {library}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode)
