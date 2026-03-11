"""Streamlit app for running and visualizing HIPPS-DIMES."""

from __future__ import annotations

import contextlib
import io
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


APP_TITLE = "HIPPS-DIMES Workbench"
APP_SUBTITLE = "Run local reconstructions, inspect matrices, and explore dynamics and mechanics without leaving Python."
INPUT_FILE_SUFFIXES = (".txt", ".csv", ".npy", ".cool", ".mcool", ".hic")
MAX_BROWSER_ENTRIES = 2000


@dataclass
class HippsBindings:
    run_optimization: Callable[..., dict[str, Any]]
    a2cmap_theory: Callable[..., np.ndarray]
    a2dmap_theory: Callable[..., np.ndarray]
    cmap2dmap: Callable[..., np.ndarray]
    cmap2dmap_missing_data: Callable[..., np.ndarray]
    compute_m1_i: Callable[..., np.ndarray]
    compute_acf_general_theory: Callable[..., tuple[np.ndarray, np.ndarray]]
    compute_modulus: Callable[..., tuple[np.ndarray, np.ndarray]]
    compute_monomer_modulus: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]]
    dmap2cmap: Callable[..., np.ndarray]
    neighbor_balance_symmetric: Callable[..., np.ndarray]
    is_gpu_available: Callable[[], bool]
    get_gpu_name: Callable[[], str | None]
    cooler: Any
    hicstraw: Any


@dataclass
class RunArtifacts:
    results: dict[str, Any]
    runtime_seconds: float
    captured_stdout: str
    config: dict[str, Any]


def _ensure_hipps_dimes_importable() -> None:
    try:
        import hipps_dimes  # noqa: F401
        return
    except ImportError:
        pass

    repo_root = Path(__file__).resolve().parents[2]
    sibling_repo = repo_root.parent / "HIPPS-DIMES"
    if sibling_repo.exists():
        sys.path.insert(0, str(sibling_repo))


@st.cache_resource(show_spinner=False)
def _load_bindings() -> HippsBindings:
    _ensure_hipps_dimes_importable()
    from hipps_dimes import (
        a2cmap_theory,
        a2dmap_theory,
        cmap2dmap,
        cmap2dmap_missing_data,
        compute_acf_general_theory,
        compute_m1_i,
        compute_modulus,
        compute_monomer_modulus,
        dmap2cmap,
        get_gpu_name,
        is_gpu_available,
        neighbor_balance_symmetric,
        run_optimization,
    )
    from hipps_dimes.numerics import cooler, hicstraw

    return HippsBindings(
        run_optimization=run_optimization,
        a2cmap_theory=a2cmap_theory,
        a2dmap_theory=a2dmap_theory,
        cmap2dmap=cmap2dmap,
        cmap2dmap_missing_data=cmap2dmap_missing_data,
        compute_m1_i=compute_m1_i,
        compute_acf_general_theory=compute_acf_general_theory,
        compute_modulus=compute_modulus,
        compute_monomer_modulus=compute_monomer_modulus,
        dmap2cmap=dmap2cmap,
        neighbor_balance_symmetric=neighbor_balance_symmetric,
        is_gpu_available=is_gpu_available,
        get_gpu_name=get_gpu_name,
        cooler=cooler,
        hicstraw=hicstraw,
    )


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 212, 163, 0.28), transparent 30%),
                radial-gradient(circle at top right, rgba(42, 157, 143, 0.16), transparent 28%),
                linear-gradient(180deg, #fffaf3 0%, #f4f1ea 100%);
        }
        .app-shell {
            border: 1px solid rgba(34, 52, 61, 0.12);
            border-radius: 24px;
            padding: 1.25rem 1.5rem 1rem 1.5rem;
            background: rgba(255, 255, 255, 0.78);
            backdrop-filter: blur(10px);
            box-shadow: 0 16px 50px rgba(32, 42, 68, 0.08);
        }
        .hero-kicker {
            letter-spacing: 0.18em;
            text-transform: uppercase;
            font-size: 0.78rem;
            font-weight: 700;
            color: #9c6644;
            margin-bottom: 0.25rem;
        }
        .hero-title {
            font-size: 2.4rem;
            font-weight: 800;
            color: #182026;
            margin-bottom: 0.3rem;
            line-height: 1.05;
        }
        .hero-subtitle {
            color: #425466;
            font-size: 1rem;
            max-width: 56rem;
        }
        .metric-card {
            border-radius: 18px;
            padding: 0.9rem 1rem;
            background: linear-gradient(145deg, rgba(255,255,255,0.95), rgba(244, 241, 234, 0.92));
            border: 1px solid rgba(34, 52, 61, 0.08);
            min-height: 110px;
        }
        .metric-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #9c6644;
            margin-bottom: 0.45rem;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #182026;
            line-height: 1.05;
        }
        .metric-note {
            color: #52606d;
            font-size: 0.92rem;
            margin-top: 0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _candidate_input_path() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    sibling_sample = repo_root.parent / "HIPPS-DIMES" / "data" / "IMR90_chr21-28-30Mb.csv"
    if sibling_sample.exists():
        return str(sibling_sample)
    return ""


def _split_browser_input_path(raw_path: str) -> str:
    return raw_path.split("::", 1)[0].strip()


def _seed_browser_directory() -> Path:
    input_path = st.session_state.get("input_path", "").strip()
    if input_path:
        candidate = Path(_split_browser_input_path(input_path)).expanduser()
        if candidate.is_file():
            return candidate.resolve().parent
        if candidate.is_dir():
            return candidate.resolve()

    sample_path = _candidate_input_path()
    if sample_path:
        return Path(sample_path).expanduser().resolve().parent
    return Path.home()


def _initialize_path_state() -> None:
    if "input_path" not in st.session_state:
        st.session_state["input_path"] = _candidate_input_path()
    if "browser_dir" not in st.session_state:
        st.session_state["browser_dir"] = str(_seed_browser_directory())
    if "browser_dir_entry" not in st.session_state:
        st.session_state["browser_dir_entry"] = st.session_state["browser_dir"]
    if "browser_notice" not in st.session_state:
        st.session_state["browser_notice"] = None


def _set_browser_directory(target: Path) -> None:
    resolved = target.expanduser()
    if resolved.is_file():
        resolved = resolved.parent
    if resolved.exists() and resolved.is_dir():
        browser_dir = str(resolved.resolve())
        st.session_state["browser_dir"] = browser_dir
        st.session_state["browser_dir_entry"] = browser_dir


def _set_browser_notice(level: str, message: str) -> None:
    st.session_state["browser_notice"] = (level, message)


def _clear_browser_notice() -> None:
    st.session_state["browser_notice"] = None


def _go_to_browser_directory() -> None:
    candidate = Path(st.session_state["browser_dir_entry"]).expanduser()
    if candidate.exists() and candidate.is_dir():
        _set_browser_directory(candidate)
        _clear_browser_notice()
    else:
        _set_browser_notice("warning", "Directory does not exist.")


def _go_up_browser_directory() -> None:
    _set_browser_directory(Path(st.session_state["browser_dir"]).expanduser().parent)
    _clear_browser_notice()


def _go_home_browser_directory() -> None:
    _set_browser_directory(Path.home())
    _clear_browser_notice()


def _sync_browser_directory_to_input() -> None:
    _set_browser_directory(_seed_browser_directory())
    _clear_browser_notice()


def _open_selected_browser_directory(selected_dir: str) -> None:
    if not selected_dir:
        _set_browser_notice("info", "Select a folder first.")
        return
    current_dir = Path(st.session_state["browser_dir"]).expanduser()
    _set_browser_directory(current_dir / selected_dir)
    _clear_browser_notice()


def _use_selected_browser_file(selected_file: str) -> None:
    if not selected_file:
        _set_browser_notice("info", "Select a file first.")
        return
    current_dir = Path(st.session_state["browser_dir"]).expanduser()
    selected_path = str((current_dir / selected_file).resolve())
    st.session_state["input_path"] = selected_path
    _set_browser_notice("success", f"Selected `{selected_file}`")


def _list_browser_entries(directory: Path, show_hidden: bool, show_all_files: bool, name_filter: str) -> tuple[list[str], list[str]]:
    directories: list[str] = []
    files: list[str] = []
    lowered_filter = name_filter.strip().lower()

    for entry in sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
        name = entry.name
        if not show_hidden and name.startswith("."):
            continue
        if lowered_filter and lowered_filter not in name.lower():
            continue

        if entry.is_dir():
            directories.append(name)
            continue

        if not entry.is_file():
            continue
        if not show_all_files and entry.suffix.lower() not in INPUT_FILE_SUFFIXES:
            continue
        files.append(name)

        if len(directories) + len(files) >= MAX_BROWSER_ENTRIES:
            break

    return directories, files


def _render_filesystem_picker() -> None:
    _initialize_path_state()
    current_dir = Path(st.session_state["browser_dir"]).expanduser()
    if not current_dir.exists() or not current_dir.is_dir():
        current_dir = _seed_browser_directory()
        _set_browser_directory(current_dir)

    st.caption("Pick a local file to populate the input path. For multires cooler files, append `::group` manually after selection.")

    notice = st.session_state.get("browser_notice")
    if notice is not None:
        level, message = notice
        getattr(st, level)(message)

    st.text_input("Directory", key="browser_dir_entry")
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
    nav_col1.button("Go", use_container_width=True, key="browser_go", on_click=_go_to_browser_directory)
    nav_col2.button("Up", use_container_width=True, key="browser_up", on_click=_go_up_browser_directory)
    nav_col3.button("Home", use_container_width=True, key="browser_home", on_click=_go_home_browser_directory)
    nav_col4.button("Sync", use_container_width=True, key="browser_sync", on_click=_sync_browser_directory_to_input)

    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1.6])
    show_hidden = filter_col1.checkbox("Hidden", value=False, key="browser_show_hidden")
    show_all_files = filter_col2.checkbox("All files", value=False, key="browser_show_all_files")
    name_filter = filter_col3.text_input("Filter", value="", key="browser_filter")

    try:
        directories, files = _list_browser_entries(current_dir, show_hidden, show_all_files, name_filter)
    except PermissionError:
        st.error(f"Permission denied: {current_dir}")
        return

    st.caption(f"Current directory: `{current_dir}`")
    dir_col, file_col = st.columns(2)
    with dir_col:
        selected_dir = st.selectbox(
            "Folders",
            options=[""] + directories,
            help="Open a folder in the browser.",
        )
        st.button(
            "Open folder",
            use_container_width=True,
            key="browser_open_dir",
            on_click=_open_selected_browser_directory,
            args=[selected_dir],
        )

    with file_col:
        selected_file = st.selectbox(
            "Files",
            options=[""] + files,
            help="Select a file and copy its absolute path into the input field.",
        )
        st.button(
            "Use file",
            use_container_width=True,
            key="browser_use_file",
            on_click=_use_selected_browser_file,
            args=[selected_file],
        )

    if not directories and not files:
        st.info("No entries matched the current filter.")


def _normalize_input_path(raw_path: str) -> str:
    raw_path = raw_path.strip()
    if not raw_path:
        return raw_path

    if "::" in raw_path:
        base_path, suffix = raw_path.split("::", 1)
        base = Path(base_path).expanduser()
        if base.exists():
            return f"{base.resolve()}::{suffix}"
        return raw_path

    path = Path(raw_path).expanduser()
    if path.exists():
        return str(path.resolve())
    return raw_path


def _parse_save_steps(raw_value: str) -> list[int] | None:
    stripped = raw_value.strip()
    if not stripped:
        return None
    steps = sorted({int(part.strip()) for part in stripped.split(",") if part.strip()})
    return steps or None


def _make_output_prefix(raw_value: str) -> str | None:
    stripped = raw_value.strip()
    if not stripped:
        return None

    path = Path(stripped).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path.resolve())


def _load_contact_map_from_source(bindings: HippsBindings, config: dict[str, Any]) -> np.ndarray:
    input_path = config["input_path"]
    input_format = config["input_format"]
    selection = config["selection"].strip()

    if input_format == "text":
        return np.loadtxt(input_path)
    if input_format == "npy":
        return np.load(input_path)
    if input_format == "cooler":
        if bindings.cooler is None:
            raise ImportError("cooler support is unavailable in this environment.")
        cmap_data = bindings.cooler.Cooler(input_path)
        cmap = cmap_data.matrix(balance=config["balance"]).fetch(selection)
        if len(cmap) >= 5000:
            raise ValueError(f"The matrix size is {len(cmap)}x{len(cmap)}. Please use a smaller matrix.")
        return np.asarray(cmap)
    if input_format == "hic":
        if bindings.hicstraw is None:
            raise ImportError(".hic support is unavailable in this environment.")
        if not selection or "," not in selection:
            raise ValueError("For .hic input, selection must be 'chr1:start1-end1,chr2:start2-end2'.")

        hic = bindings.hicstraw.HiCFile(input_path)
        reg1, reg2 = selection.split(",")
        raw_chrom1, region1 = reg1.split(":")
        raw_chrom2, region2 = reg2.split(":")
        chrom1 = raw_chrom1[3:] if raw_chrom1.lower().startswith("chr") else raw_chrom1
        chrom2 = raw_chrom2[3:] if raw_chrom2.lower().startswith("chr") else raw_chrom2
        start1, end1 = map(int, region1.split("-"))
        start2, end2 = map(int, region2.split("-"))

        matrix_obj = hic.getMatrixZoomData(chrom1, chrom2, "observed", config["hic_norm"], config["hic_unit"], config["binsize"])
        try:
            return np.asarray(matrix_obj.getRecordsAsMatrix(start1, end1, start2, end2))
        except Exception:
            region1 = f"{chrom1}:{start1}:{end1}"
            region2 = f"{chrom2}:{start2}:{end2}"
            result = bindings.hicstraw.straw(
                "observed",
                config["hic_norm"],
                input_path,
                region1,
                region2,
                config["hic_unit"],
                config["binsize"],
            )
            dim1 = (end1 - start1) // config["binsize"] + 1
            dim2 = (end2 - start2) // config["binsize"] + 1
            cmap = np.zeros((dim1, dim2))
            for pt in result:
                i = int((pt.binX - start1) / config["binsize"])
                j = int((pt.binY - start2) / config["binsize"])
                cmap[i, j] = pt.counts
            return cmap + cmap.T

    raise ValueError(f"Unsupported input format for contact map target: {input_format}")


def _normalize_contact_map(cmap: np.ndarray) -> np.ndarray:
    cmap = np.asarray(cmap, dtype=float)
    max_value = np.nanmax(cmap)
    if not np.isfinite(max_value) or max_value == 0.0:
        return cmap
    return cmap / max_value


def _build_target_matrices(bindings: HippsBindings, config: dict[str, Any]) -> dict[str, np.ndarray]:
    input_path = config["input_path"]
    input_type = config["input_type"]
    input_format = config["input_format"]

    if input_type == "dmap":
        if input_format == "text":
            dmap_target = np.loadtxt(input_path)
        elif input_format == "npy":
            dmap_target = np.load(input_path)
        else:
            raise ValueError("input_type='dmap' only supports input_format='text' or 'npy'.")
        return {"dmap_target": np.asarray(dmap_target, dtype=float)}

    if input_type == "ddmap":
        if input_format == "text":
            ddmap_target = np.loadtxt(input_path)
        elif input_format == "npy":
            ddmap_target = np.load(input_path)
        else:
            raise ValueError("input_type='ddmap' only supports input_format='text' or 'npy'.")
        dmap_target = np.sqrt((8.0 / (3.0 * np.pi)) * np.asarray(ddmap_target, dtype=float))
        return {"dmap_target": dmap_target}

    if input_type == "cmap":
        cmap_target = _load_contact_map_from_source(bindings, config)
        if config["neighbor_balance"]:
            cmap_target = bindings.neighbor_balance_symmetric(
                cmap_target,
                not_normalize=config["not_normalize"],
            )

        if config["ignore_missing_data"]:
            dmap_target = bindings.cmap2dmap_missing_data(
                cmap_target,
                config["alpha"],
                config["not_normalize"],
            )
        else:
            dmap_target = bindings.cmap2dmap(
                cmap_target,
                config["alpha"],
                config["not_normalize"],
            )

        return {
            "dmap_target": np.asarray(dmap_target, dtype=float),
            "cmap_target": _normalize_contact_map(cmap_target),
        }

    return {}


def _combine_triangle_matrices(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape != upper.shape:
        raise ValueError(f"Cannot compare matrices with different shapes: {lower.shape} vs {upper.shape}")

    combined = np.zeros_like(upper, dtype=float)
    combined += np.tril(lower, k=-1)
    combined += np.triu(upper, k=1)
    diagonal = np.diag(upper)
    if diagonal.size:
        np.fill_diagonal(combined, diagonal)
    return combined


def _render_header(gpu_summary: str) -> None:
    st.markdown(
        f"""
        <div class="app-shell">
          <div class="hero-kicker">Local Workbench</div>
          <div class="hero-title">{APP_TITLE}</div>
          <div class="hero-subtitle">{APP_SUBTITLE}<br><br><strong>Compute backend:</strong> {gpu_summary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(bindings: HippsBindings) -> dict[str, Any] | None:
    gpu_available = bindings.is_gpu_available()
    gpu_name = bindings.get_gpu_name() if gpu_available else None

    with st.sidebar:
        _initialize_path_state()
        st.header("Run Setup")
        if gpu_available:
            st.caption(f"GPU available: {gpu_name}")
        else:
            st.caption("GPU unavailable in this environment. CPU mode will be used.")

        st.text_input(
            "Input file path",
            key="input_path",
            help="Absolute path to a local matrix or Hi-C file. Keep this editable for multires `::group` suffixes.",
        )
        with st.expander("Browse local files", expanded=False):
            _render_filesystem_picker()

        with st.form("run_form", clear_on_submit=False):
            output_prefix = st.text_input(
                "Output prefix (optional)",
                value="",
                help="If provided, HIPPS-DIMES writes the standard output files to this prefix.",
            )

            input_col, format_col = st.columns(2)
            input_type = input_col.selectbox("Input type", ["cmap", "dmap", "ddmap"], index=0)
            input_format = format_col.selectbox("Input format", ["text", "npy", "cooler", "hic"], index=0)

            selection = st.text_input(
                "Selection / region",
                value="",
                help="Required for cooler and .hic inputs.",
            )

            method_col, ensemble_col = st.columns(2)
            method = method_col.selectbox("Method", ["IS", "GD", "DI"], index=0)
            ensemble = ensemble_col.number_input("Ensemble size", min_value=1, value=100, step=10)

            iter_col, alpha_col = st.columns(2)
            iteration = iter_col.number_input("Iterations", min_value=1, value=5000, step=500)
            alpha = alpha_col.number_input("Alpha", min_value=0.1, value=4.0, step=0.1)

            learning_rate = st.number_input("Learning rate", value=10.0, step=0.5)

            with st.expander("Acceleration", expanded=True):
                momentum_col, nesterov_col = st.columns(2)
                momentum = momentum_col.number_input("Momentum", min_value=0.0, max_value=1.0, value=0.95, step=0.05)
                nesterov = nesterov_col.checkbox("Use Nesterov", value=True)
                gpu_col, gpu32_col = st.columns(2)
                use_gpu = gpu_col.checkbox("Use GPU", value=gpu_available, disabled=not gpu_available)
                gpu_float32 = gpu32_col.checkbox("GPU float32", value=False, disabled=not gpu_available)
                eigh_threads = st.number_input(
                    "Eigh / BLAS threads",
                    min_value=0,
                    value=0,
                    help="Use 0 for backend default.",
                )

            with st.expander("Regularization and noise"):
                lamd = st.number_input(
                    "Lambda",
                    min_value=0.0,
                    value=0.0,
                    step=1e-4,
                    format="%.6g",
                )
                reg = st.selectbox("Regularization", ["L2", "L1"], index=0)
                gaussian_noise_variance = st.number_input(
                    "Gaussian noise variance",
                    min_value=0.0,
                    value=0.0,
                    step=0.01,
                )

            with st.expander("Format-specific options"):
                binsize_col, norm_col = st.columns(2)
                binsize = binsize_col.number_input("Hi-C binsize", min_value=1, value=25000, step=1000)
                hic_norm = norm_col.selectbox("Hi-C norm", ["KR", "VC", "NONE"], index=0)
                unit_col, balance_col = st.columns(2)
                hic_unit = unit_col.selectbox("Hi-C unit", ["BP", "FRAG"], index=0)
                balance = balance_col.checkbox("Cooler balance", value=False)
                neighbor_balance = st.checkbox("Neighbor balance", value=False)
                not_normalize = st.checkbox("Skip contact-map normalization", value=False)

            with st.expander("Output and robustness"):
                save_steps = st.text_input("Save steps", value="", help="Comma-separated iterations.")
                no_log = st.checkbox("Skip CSV logs", value=False)
                no_xyzs = st.checkbox("Skip XYZ generation", value=False)
                ignore_missing_data = st.checkbox("Ignore missing data", value=False)
                enforce_nonnegative = st.checkbox("Enforce nonnegative springs", value=False)

            run_clicked = st.form_submit_button("Run HIPPS-DIMES", use_container_width=True)

        if not run_clicked:
            return None

        config = {
            "input_path": st.session_state.get("input_path", ""),
            "output_prefix": output_prefix,
            "input_type": input_type,
            "input_format": input_format,
            "selection": selection,
            "method": method,
            "ensemble": int(ensemble),
            "iteration": int(iteration),
            "alpha": float(alpha),
            "learning_rate": float(learning_rate),
            "momentum": float(momentum),
            "nesterov": bool(nesterov),
            "use_gpu": bool(use_gpu),
            "gpu_float32": bool(gpu_float32),
            "eigh_threads": None if int(eigh_threads) == 0 else int(eigh_threads),
            "lamd": float(lamd),
            "reg": reg,
            "gaussian_noise_variance": float(gaussian_noise_variance),
            "binsize": int(binsize),
            "hic_norm": hic_norm,
            "hic_unit": hic_unit,
            "balance": bool(balance),
            "neighbor_balance": bool(neighbor_balance),
            "not_normalize": bool(not_normalize),
            "save_steps": save_steps,
            "no_log": bool(no_log),
            "no_xyzs": bool(no_xyzs),
            "ignore_missing_data": bool(ignore_missing_data),
            "enforce_nonnegative_connectivity_matrix": bool(enforce_nonnegative),
        }
        return config


def _validate_config(config: dict[str, Any]) -> tuple[bool, str | None]:
    if not config["input_path"].strip():
        return False, "An input file path is required."
    if config["input_format"] in {"cooler", "hic"} and not config["selection"].strip():
        return False, "Selection is required for cooler and .hic inputs."
    if config["gaussian_noise_variance"] > 0.0 and config["lamd"] > 0.0:
        return False, "Gaussian noise variance cannot be combined with lambda regularization."
    return True, None


def _run_model(bindings: HippsBindings, config: dict[str, Any]) -> RunArtifacts:
    normalized_path = _normalize_input_path(config["input_path"])
    output_prefix = _make_output_prefix(config["output_prefix"])
    save_steps = _parse_save_steps(config["save_steps"])

    kwargs = {
        "input_path": normalized_path,
        "output_prefix": output_prefix,
        "ensemble": config["ensemble"],
        "alpha": config["alpha"],
        "selection": config["selection"].strip() or None,
        "method": config["method"],
        "lamd": config["lamd"],
        "reg": config["reg"],
        "gaussian_noise_variance": config["gaussian_noise_variance"],
        "iteration": config["iteration"],
        "learning_rate": config["learning_rate"],
        "momentum": config["momentum"],
        "nesterov": config["nesterov"],
        "use_gpu": config["use_gpu"],
        "gpu_float32": config["gpu_float32"],
        "input_type": config["input_type"],
        "input_format": config["input_format"],
        "binsize": config["binsize"],
        "hic_norm": config["hic_norm"],
        "hic_unit": config["hic_unit"],
        "no_log": config["no_log"],
        "no_xyzs": config["no_xyzs"],
        "ignore_missing_data": config["ignore_missing_data"],
        "balance": config["balance"],
        "not_normalize": config["not_normalize"],
        "neighbor_balance": config["neighbor_balance"],
        "enforce_nonnegative_connectivity_matrix": config["enforce_nonnegative_connectivity_matrix"],
        "save_steps": save_steps,
        "eigh_threads": config["eigh_threads"],
        "verbose": False,
    }

    stdout_buffer = io.StringIO()
    start = time.perf_counter()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stdout_buffer):
        results = bindings.run_optimization(**kwargs)
    runtime_seconds = time.perf_counter() - start

    try:
        results.update(_build_target_matrices(bindings, config))
    except Exception as exc:
        results["matrix_target_error"] = str(exc)

    if config["input_type"] == "cmap" and "rc_optimal" in results:
        results["cmap_final"] = bindings.dmap2cmap(
            bindings.a2dmap_theory(
                results["connectivity_matrix"],
                force_positive_definite=True,
            ),
            results["rc_optimal"],
        )

    config_with_paths = dict(config)
    config_with_paths["input_path"] = normalized_path
    config_with_paths["output_prefix"] = output_prefix
    config_with_paths["save_steps"] = save_steps

    return RunArtifacts(
        results=results,
        runtime_seconds=runtime_seconds,
        captured_stdout=stdout_buffer.getvalue().strip(),
        config=config_with_paths,
    )


def _metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _plot_matrix(
    matrix: np.ndarray,
    title: str,
    colorscale: str,
    midpoint: float | None = None,
    log_transform: bool = False,
) -> go.Figure:
    plot_matrix = np.asarray(matrix)
    colorbar_title = "value"
    if log_transform:
        plot_matrix = np.where(plot_matrix > 0, np.log(plot_matrix), np.nan)
        colorbar_title = "log(value)"

    figure = go.Figure(
        data=[
            go.Heatmap(
                z=plot_matrix,
                colorscale=colorscale,
                zmid=midpoint,
                colorbar=dict(title=colorbar_title),
                hovertemplate="row=%{y}<br>col=%{x}<br>value=%{z:.4g}<extra></extra>",
            )
        ]
    )
    figure.update_layout(
        title=title,
        height=620,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Locus",
        yaxis_title="Locus",
    )
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    figure.update_yaxes(autorange="reversed")
    return figure


def _plot_convergence(iteration_series: pd.DataFrame) -> go.Figure:
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=iteration_series["iteration"],
            y=iteration_series["loss"],
            mode="lines",
            line=dict(color="#d97706", width=2.5),
            name="Loss",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=iteration_series["iteration"],
            y=iteration_series["entropy"],
            mode="lines",
            line=dict(color="#0f766e", width=2.5),
            name="Entropy",
        ),
        secondary_y=True,
    )
    figure.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20))
    figure.update_xaxes(title_text="Iteration", type="log")
    figure.update_yaxes(title_text="Loss", secondary_y=False)
    figure.update_yaxes(title_text="Entropy", secondary_y=True)
    return figure


def _plot_structure(coords: np.ndarray) -> go.Figure:
    centered = coords - coords.mean(axis=0, keepdims=True)
    indices = np.arange(centered.shape[0])
    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=centered[:, 0],
                y=centered[:, 1],
                z=centered[:, 2],
                mode="lines+markers",
                marker=dict(
                    size=4,
                    color=indices,
                    colorscale="Turbo",
                    colorbar=dict(title="Locus"),
                ),
                line=dict(color="#1f2937", width=5),
                hovertemplate="locus=%{marker.color}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
            )
        ]
    )
    figure.update_layout(
        height=700,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return figure


def _plot_two_series(
    x1: np.ndarray,
    y1: np.ndarray,
    label1: str,
    x2: np.ndarray | None = None,
    y2: np.ndarray | None = None,
    label2: str | None = None,
    log_x: bool = True,
    log_y: bool = True,
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x1,
            y=y1,
            mode="lines",
            line=dict(color="#d97706", width=2.5),
            name=label1,
        )
    )
    if x2 is not None and y2 is not None and label2 is not None:
        figure.add_trace(
            go.Scatter(
                x=x2,
                y=y2,
                mode="lines",
                line=dict(color="#0f766e", width=2.5),
                name=label2,
            )
        )
    figure.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20))
    figure.update_xaxes(title_text="Scale", type="log" if log_x else "linear")
    figure.update_yaxes(type="log" if log_y else "linear")
    return figure


def _render_overview(artifacts: RunArtifacts) -> None:
    results = artifacts.results
    iteration_series = results["iteration_series"]
    connectivity_matrix = results["connectivity_matrix"]
    run_parameters = results["run_parameters"]
    final_loss = iteration_series["loss"].iloc[-1] if not iteration_series.empty else np.nan
    final_entropy = iteration_series["entropy"].iloc[-1] if not iteration_series.empty else np.nan

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _metric_card("Matrix size", str(connectivity_matrix.shape[0]), "Loci in the final connectivity matrix.")
    with col2:
        _metric_card("Runtime", f"{artifacts.runtime_seconds:.2f}s", "Wall-clock time inside the Streamlit run.")
    with col3:
        _metric_card("Final loss", f"{final_loss:.4g}", "Last point from the iteration series.")
    with col4:
        _metric_card("Final entropy", f"{final_entropy:.4g}", "Entropy reported by HIPPS-DIMES.")

    st.markdown("")
    left, right = st.columns([1.25, 1.0])
    with left:
        st.subheader("Run parameters")
        st.dataframe(run_parameters, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Artifacts")
        lines = [
            f"Input: `{artifacts.config['input_path']}`",
            f"Output prefix: `{artifacts.config['output_prefix'] or 'in-memory only'}`",
            f"Saved XYZs: `{'xyzs' in results}`",
            f"Final contact map: `{'cmap_final' in results}`",
            f"Intermediate checkpoints: `{len(results.get('connectivity_matrix_at_steps', {}))}`",
        ]
        st.markdown("\n".join(f"- {line}" for line in lines))

        if artifacts.captured_stdout:
            with st.expander("Captured stdout"):
                st.code(artifacts.captured_stdout)

        st.download_button(
            "Download iteration series CSV",
            data=results["iteration_series"].to_csv(index=False),
            file_name="iteration_series.csv",
            mime="text/csv",
            use_container_width=True,
        )

        matrix_buffer = io.StringIO()
        np.savetxt(matrix_buffer, connectivity_matrix)
        st.download_button(
            "Download connectivity matrix",
            data=matrix_buffer.getvalue(),
            file_name="connectivity_matrix.txt",
            mime="text/plain",
            use_container_width=True,
        )


def _render_matrices(results: dict[str, Any]) -> None:
    if "matrix_target_error" in results:
        st.warning(f"Target matrix reconstruction failed: {results['matrix_target_error']}")

    distance_matrix = results["dmap_final"]
    distance_title = "Final distance map"
    if "dmap_target" in results:
        distance_matrix = _combine_triangle_matrices(results["dmap_target"], results["dmap_final"])
        distance_title = "Distance map comparison"

    options: dict[str, tuple[np.ndarray, str, float | None, bool, str]] = {
        "Final distance map": (distance_matrix, "Sunset", None, False, distance_title),
        "Connectivity matrix": (results["connectivity_matrix"], "Tealrose", 0.0, False, "Connectivity matrix"),
    }
    if "cmap_final" in results and "cmap_target" in results:
        options["Final contact map"] = (
            _combine_triangle_matrices(results["cmap_target"], results["cmap_final"]),
            "Viridis",
            None,
            True,
            "Contact map comparison",
        )
    elif "cmap_final" in results:
        options["Final contact map"] = (results["cmap_final"], "Viridis", None, True, "Final contact map")

    labels = list(options.keys())
    selected = st.selectbox("Matrix", labels, index=0)
    matrix, colorscale, midpoint, log_transform, title = options[selected]
    st.plotly_chart(
        _plot_matrix(matrix, title, colorscale, midpoint, log_transform=log_transform),
        use_container_width=True,
    )

    if selected == "Final distance map" and "dmap_target" in results:
        st.caption("Lower triangle: target dmap. Upper triangle: HIPPS-DIMES result.")
    if selected == "Final contact map" and "cmap_target" in results:
        st.caption("Lower triangle: target cmap. Upper triangle: HIPPS-DIMES result.")

    if "connectivity_matrix_at_steps" in results:
        checkpoints = results["connectivity_matrix_at_steps"]
        if checkpoints:
            st.subheader("Intermediate connectivity checkpoints")
            selected_step = st.selectbox(
                "Checkpoint iteration",
                sorted(checkpoints.keys()),
                format_func=lambda value: f"Iteration {value}",
            )
            st.plotly_chart(
                _plot_matrix(checkpoints[selected_step], f"Connectivity at iteration {selected_step}", "Tealrose", 0.0),
                use_container_width=True,
            )


def _render_convergence(results: dict[str, Any]) -> None:
    iteration_series = results["iteration_series"]
    if iteration_series.empty:
        st.info("No iteration-series data returned.")
        return
    st.plotly_chart(_plot_convergence(iteration_series), use_container_width=True)
    st.dataframe(iteration_series.tail(25), use_container_width=True, hide_index=True)


def _render_structures(results: dict[str, Any]) -> None:
    xyzs = results.get("xyzs")
    if xyzs is None:
        st.info("XYZ generation was disabled for this run.")
        return

    snapshot_index = st.slider(
        "Structure snapshot",
        min_value=0,
        max_value=int(xyzs.shape[0] - 1),
        value=0,
    )
    st.plotly_chart(_plot_structure(xyzs[snapshot_index]), use_container_width=True)

    radius_of_gyration = np.sqrt(np.mean(np.sum((xyzs[snapshot_index] - xyzs[snapshot_index].mean(axis=0)) ** 2, axis=1)))
    st.caption(f"Snapshot {snapshot_index} radius of gyration: {radius_of_gyration:.4f}")


def _render_dynamics(bindings: HippsBindings, results: dict[str, Any]) -> None:
    connectivity_matrix = results["connectivity_matrix"]
    n_loci = connectivity_matrix.shape[0]
    analysis = st.radio(
        "Dynamics analysis",
        ["Single-locus MSD", "Pair ACF + 2-point MSD"],
        horizontal=True,
    )

    col1, col2, col3 = st.columns(3)
    t_min_exp = col1.number_input("Min log10(t)", value=-3.0, step=0.5, key="dyn_t_min")
    t_max_exp = col2.number_input("Max log10(t)", value=3.0, step=0.5, key="dyn_t_max")
    points = col3.number_input("Points", min_value=20, value=200, step=20, key="dyn_points")
    zeta = st.number_input("Zeta", min_value=0.01, value=1.0, step=0.1, key="dyn_zeta")
    times = np.logspace(t_min_exp, t_max_exp, int(points))

    if analysis == "Single-locus MSD":
        locus = st.slider("Locus", min_value=0, max_value=int(n_loci - 1), value=0)
        msd = bindings.compute_m1_i(locus, times, connectivity_matrix, zeta=zeta)
        figure = _plot_two_series(msd[:, 0], msd[:, 1], f"MSD locus {locus}")
        figure.update_xaxes(title_text="Time")
        figure.update_yaxes(title_text="MSD")
        st.plotly_chart(figure, use_container_width=True)
        st.dataframe(pd.DataFrame(msd, columns=["time", "msd"]), use_container_width=True, hide_index=True)
    else:
        i_col, j_col = st.columns(2)
        i = i_col.slider("Locus i", min_value=0, max_value=int(n_loci - 1), value=0)
        j = j_col.slider("Locus j", min_value=0, max_value=int(n_loci - 1), value=min(1, n_loci - 1))
        acf, two_point_msd = bindings.compute_acf_general_theory(i, j, times, connectivity_matrix, zeta=zeta)
        figure = _plot_two_series(
            acf[:, 0],
            acf[:, 1],
            f"ACF ({i}, {j})",
            two_point_msd[:, 0],
            two_point_msd[:, 1],
            f"2-point MSD ({i}, {j})",
            log_x=True,
            log_y=False,
        )
        figure.update_xaxes(title_text="Time")
        figure.update_yaxes(title_text="Value")
        st.plotly_chart(figure, use_container_width=True)
        merged = pd.DataFrame(
            {
                "time": acf[:, 0],
                "acf": acf[:, 1],
                "two_point_msd": two_point_msd[:, 1],
            }
        )
        st.dataframe(merged, use_container_width=True, hide_index=True)


def _render_mechanics(bindings: HippsBindings, results: dict[str, Any]) -> None:
    connectivity_matrix = results["connectivity_matrix"]
    n_loci = connectivity_matrix.shape[0]

    col1, col2, col3 = st.columns(3)
    f_min_exp = col1.number_input("Min log10(freq)", value=-3.0, step=0.5, key="mech_f_min")
    f_max_exp = col2.number_input("Max log10(freq)", value=3.0, step=0.5, key="mech_f_max")
    points = col3.number_input("Points", min_value=20, value=200, step=20, key="mech_points")
    zeta = st.number_input("Zeta", min_value=0.01, value=1.0, step=0.1, key="mech_zeta")

    freq = np.logspace(f_min_exp, f_max_exp, int(points))
    g_storage, g_loss = bindings.compute_modulus(connectivity_matrix, freq, zeta=zeta)
    locus = st.slider("Locus for per-locus modulus", min_value=0, max_value=int(n_loci - 1), value=0)
    freq_out, g_prime_i, g_double_prime_i = bindings.compute_monomer_modulus(connectivity_matrix, freq, zeta=zeta)

    st.subheader("Bulk moduli")
    bulk_figure = _plot_two_series(
        g_storage[:, 0],
        g_storage[:, 1],
        "G' (storage)",
        g_loss[:, 0],
        g_loss[:, 1],
        "G'' (loss)",
    )
    bulk_figure.update_xaxes(title_text="Angular frequency")
    bulk_figure.update_yaxes(title_text="Modulus")
    st.plotly_chart(bulk_figure, use_container_width=True)

    st.subheader(f"Per-locus modulus: locus {locus}")
    locus_figure = _plot_two_series(
        freq_out,
        g_prime_i[:, locus],
        f"G' locus {locus}",
        freq_out,
        g_double_prime_i[:, locus],
        f"G'' locus {locus}",
    )
    locus_figure.update_xaxes(title_text="Angular frequency")
    locus_figure.update_yaxes(title_text="Modulus")
    st.plotly_chart(locus_figure, use_container_width=True)


def _render_results(bindings: HippsBindings, artifacts: RunArtifacts) -> None:
    tabs = st.tabs(["Overview", "Matrices", "Convergence", "3D Structure", "Dynamics", "Mechanics"])
    with tabs[0]:
        _render_overview(artifacts)
    with tabs[1]:
        _render_matrices(artifacts.results)
    with tabs[2]:
        _render_convergence(artifacts.results)
    with tabs[3]:
        _render_structures(artifacts.results)
    with tabs[4]:
        _render_dynamics(bindings, artifacts.results)
    with tabs[5]:
        _render_mechanics(bindings, artifacts.results)


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()

    try:
        bindings = _load_bindings()
    except Exception as exc:
        st.title(APP_TITLE)
        st.error("HIPPS-DIMES could not be imported.")
        st.code(
            "uv pip install -e ../HIPPS-DIMES\n"
            "uv pip install -e .\n"
            "streamlit run app.py"
        )
        st.exception(exc)
        return

    gpu_summary = bindings.get_gpu_name() if bindings.is_gpu_available() else "CPU only"
    _render_header(gpu_summary)
    st.markdown("")

    config = _render_sidebar(bindings)
    if config is not None:
        valid, error_message = _validate_config(config)
        if not valid:
            st.error(error_message)
        else:
            with st.spinner("Running HIPPS-DIMES..."):
                try:
                    st.session_state["artifacts"] = _run_model(bindings, config)
                except Exception as exc:
                    st.session_state.pop("artifacts", None)
                    st.exception(exc)

    artifacts = st.session_state.get("artifacts")
    if artifacts is None:
        st.info(
            "Configure a run in the sidebar, then execute HIPPS-DIMES. "
            "The app will keep the most recent result in memory for interactive inspection."
        )
        return

    _render_results(bindings, artifacts)
