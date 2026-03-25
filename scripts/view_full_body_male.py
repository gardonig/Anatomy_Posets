from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons


def load_volume_from_png_folder(folder: Path) -> np.ndarray:
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    png_paths = sorted(folder.glob("*.png"))
    if not png_paths:
        raise RuntimeError(f"No .png files found in {folder}")

    slices: list[np.ndarray] = []
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    for p in png_paths:
        try:
            img = Image.open(p)
            img.load()
            # Preserve RGB information instead of converting to grayscale.
            img = img.convert("RGB")
            arr = np.array(img, dtype=np.float32)
        except OSError as exc:
            print(f"Skipping corrupt/truncated image {p.name}: {exc}")
            continue
        slices.append(arr)

    if not slices:
        raise RuntimeError(f"All .png files in {folder} are corrupt or unreadable.")

    # (Z, Y, X, 3) RGB volume
    volume = np.stack(slices, axis=0).astype(np.float32)  # (Z, Y, X, 3)

    vmin = volume.min()
    vmax = volume.max()
    if vmax > vmin:
        # Normalize to [0, 1]
        volume = (volume - vmin) / (vmax - vmin)
    else:
        volume = np.zeros_like(volume, dtype=np.float32)

    print(volume.shape)
    return volume


def show_volume_single_plane(volume: np.ndarray) -> None:
    # volume is (Z, Y, X, 3)
    if volume.ndim != 4 or volume.shape[-1] != 3:
        raise ValueError(f"Expected RGB volume of shape (Z, Y, X, 3), got {volume.shape!r}")

    z_dim, y_dim, x_dim, _ = volume.shape
    dims = {"axial": z_dim, "coronal": y_dim, "sagittal": x_dim}
    current_plane = "axial"

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(left=0.25, bottom=0.15, right=0.98, top=0.95)

    def get_extent(plane: str) -> tuple[float, float, float, float]:
        # Axial: pixels are 1mm x 1mm, no scaling.
        if plane == "axial":
            return (0.0, float(x_dim), 0.0, float(y_dim))
        # Coronal: Z axis is 3x thicker than X.
        if plane == "coronal":
            return (0.0, float(x_dim), 0.0, float(z_dim) * 1.0)
        # Sagittal: Z axis is 3x thicker than Y.
        return (0.0, float(y_dim), 0.0, float(z_dim) * 1.0)

    def get_slice(plane: str, idx: int) -> np.ndarray:
        if plane == "axial":
            idx = max(0, min(idx, z_dim - 1))
            # axial: no flip (Y, X, 3)
            return volume[idx, :, :, :]
        if plane == "coronal":
            idx = max(0, min(idx, y_dim - 1))
            sl = volume[:, idx, :, :]  # (Z, X, 3)
            # coronal: flip vertically
            return np.flipud(sl)
        idx = max(0, min(idx, x_dim - 1))
        sl = volume[:, :, idx, :]  # (Z, Y, 3)
        return np.flipud(sl)

    init_idx = z_dim // 2
    img = ax.imshow(
        get_slice(current_plane, init_idx),
        origin="lower",
        extent=get_extent(current_plane),
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{current_plane.capitalize()} slice {init_idx}")

    ax_slider = plt.axes([0.25, 0.05, 0.7, 0.03])
    max_dim = max(z_dim, y_dim, x_dim)
    s_idx = Slider(ax_slider, "Index", 0, max_dim - 1, valinit=init_idx, valfmt="%0.0f")

    ax_radio = plt.axes([0.05, 0.4, 0.15, 0.2])
    radio = RadioButtons(ax_radio, ("axial", "coronal", "sagittal"), active=0)

    def update_slice(val: float) -> None:
        idx = int(round(s_idx.val))
        plane = current_plane
        sl = get_slice(plane, idx)
        img.set_data(sl)
        img.set_extent(get_extent(plane))
        ax.set_title(f"{plane.capitalize()} slice {idx}")
        fig.canvas.draw_idle()

    def update_plane(label: str) -> None:
        nonlocal current_plane
        current_plane = label
        update_slice(s_idx.val)

    s_idx.on_changed(update_slice)
    radio.on_clicked(update_plane)

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=Path,
        default=None,
        help="Folder containing downsampled full-body PNG slices (default: ../full_body_male/downsampled)",
    )
    parser.add_argument(
        "--mode",
        choices=("new", "old"),
        default="new",
        help=(
            '"new": rebuild volume from PNGs (optionally saving .npy); '
            '"old": load the last saved .npy volume and just view it (no rebuilding).'
        ),
    )
    parser.add_argument(
        "--crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Crop the volume to remove empty margins for size efficiency (default: true). "
            "Uses coronal(Y)=33..322 and sagittal(X)=55..624 in the ORIGINAL tensor axes."
        ),
    )
    parser.add_argument(
        "--save-npy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the loaded volume as .npy tensors (default: true). Use --no-save-npy to skip saving.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent

    if args.mode == "new":
        if args.folder is None:
            folder = (repo_root / ".." / "visible_human" / "full_body_male" / "downsampled").resolve()
        else:
            folder = args.folder.resolve()

        print(f"[mode=new] Loading volume from PNG folder: {folder}")
        volume = load_volume_from_png_folder(folder)
        print(f"RGB volume shape (Z, Y, X, 3): {volume.shape}")

        if args.crop:
            z_dim, y_dim, x_dim, _ = volume.shape
            # Coronal (Y): keep 33..322 → 290 slices after cropping.
            y0, y1 = 33, 322
            x0, x1 = 55, 624
            # Clamp to volume bounds, keep inclusive end indices.
            y0 = max(0, min(y0, y_dim - 1))
            y1 = max(0, min(y1, y_dim - 1))
            x0 = max(0, min(x0, x_dim - 1))
            x1 = max(0, min(x1, x_dim - 1))
            if y1 < y0:
                y0, y1 = y1, y0
            if x1 < x0:
                x0, x1 = x1, x0
            volume = volume[:, y0 : y1 + 1, x0 : x1 + 1, :]
            print(
                f"[mode=new] Cropped volume to Y={y0}..{y1}, X={x0}..{x1} -> shape {volume.shape}"
            )

        if args.save_npy:
            # Save RGB 4D tensor as the primary volume.
            tensors_dir = (repo_root / "assets" / "visible_human_tensors")
            tensors_dir.mkdir(parents=True, exist_ok=True)
            out_path_rgb = tensors_dir / "full_body_tensor_rgb.npy"
            np.save(out_path_rgb, volume)
            print(f"Saved RGB volume tensor to: {out_path_rgb}")

            # Also save a grayscale 3D tensor (Z, Y, X) for legacy consumers if needed.
            gray_volume = volume[..., 0]
            out_path_gray = tensors_dir / "full_body_tensor.npy"
            np.save(out_path_gray, gray_volume)
            print(f"Saved grayscale volume tensor to: {out_path_gray}")
        else:
            print("[mode=new] Skipping .npy save (use --save-npy to enable).")
    else:
        # mode == "old": load last saved tensor from assets/visible_human_tensors, prefer RGB.
        tensors_dir = (repo_root / "assets" / "visible_human_tensors")
        rgb_path = tensors_dir / "full_body_tensor_rgb.npy"
        gray_path = tensors_dir / "full_body_tensor.npy"
        if rgb_path.exists():
            print(f"[mode=old] Loading RGB volume from: {rgb_path}")
            volume = np.load(rgb_path)
        elif gray_path.exists():
            print(f"[mode=old] Loading grayscale volume from: {gray_path}")
            gray_vol = np.load(gray_path)
            if gray_vol.ndim != 3:
                raise ValueError(f"Expected grayscale tensor (Z, Y, X), got shape {gray_vol.shape!r}")
            # Promote grayscale to RGB for the viewer.
            volume = np.stack([gray_vol, gray_vol, gray_vol], axis=-1)
        else:
            raise SystemExit(
                "No existing volume found for mode='old'. "
                "Expected full_body_tensor_rgb.npy or full_body_tensor.npy under assets/visible_human_tensors. "
                "Run with --mode new first to build them."
            )

    show_volume_single_plane(volume)


if __name__ == "__main__":
    main()

