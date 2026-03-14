"""Streamlit app for running and visualizing HIPPS-DIMES."""

from __future__ import annotations

import contextlib
import io
import inspect
import json
import pickle
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


APP_TITLE = "HIPPS-DIMES Workbench"
APP_SUBTITLE = "Run local reconstructions, inspect matrices, and explore dynamics and mechanics without leaving Python."
INPUT_FILE_SUFFIXES = (".txt", ".csv", ".npy", ".cool", ".mcool", ".hic")
MAX_BROWSER_ENTRIES = 2000
COOLER_FILE_SUFFIXES = {".cool", ".mcool"}
RESULT_FILE_SUFFIXES = (
    "_connectivity_matrix.txt",
    "_dmap_final.txt",
    "_cmap_final.txt",
    "_cmap_target.txt",
    "_iteration_series.csv",
    "_run_parameters.csv",
    ".xyz",
)
RESULT_PICKLE_SUFFIXES = (".pkl", ".pickle")
ITERATION_SERIES_COLUMNS = ["iteration", "loss", "entropy"]
SUPPORTED_INPUT_FORMATS = {
    "cmap": ("text", "npy", "cooler", "hic"),
    "dmap": ("text", "npy"),
    "ddmap": ("text", "npy"),
}


@dataclass
class HippsBindings:
    run_optimization: Callable[..., dict[str, Any]]
    a2xyz_sample: Callable[..., np.ndarray]
    cmap2dmap: Callable[..., np.ndarray]
    cmap2dmap_missing_data: Callable[..., np.ndarray]
    compute_m1_i: Callable[..., np.ndarray]
    compute_acf_general_theory: Callable[..., tuple[np.ndarray, np.ndarray]]
    compute_modulus: Callable[..., tuple[np.ndarray, np.ndarray]]
    compute_monomer_modulus: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]]
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


class _StreamlitOutputBuffer(io.TextIOBase):
    _ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    _PROGRESS_RE = re.compile(
        r"(?P<current>\d+)/(?P<total>\d+).*?loss=(?P<loss>[-+0-9.eE]+|nan|inf).*?entropy=(?P<entropy>[-+0-9.eE]+|nan|inf)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        placeholder: Any | None = None,
        *,
        progress_bar_placeholder: Any | None = None,
        progress_summary_placeholder: Any | None = None,
        entropy_chart_placeholder: Any | None = None,
        max_lines: int = 18,
        update_interval: float = 0.25,
    ) -> None:
        self._placeholder = placeholder
        self._progress_bar_placeholder = progress_bar_placeholder
        self._progress_summary_placeholder = progress_summary_placeholder
        self._entropy_chart_placeholder = entropy_chart_placeholder
        self._max_lines = max_lines
        self._update_interval = update_interval
        self._chart_update_interval = 0.75
        self._full_buffer = io.StringIO()
        self._history_lines: list[str] = []
        self._current_line = ""
        self._last_render = 0.0
        self._last_chart_render = 0.0
        self._progress_points: list[tuple[int, float]] = []
        self._loss_points: list[tuple[int, float | None]] = []
        self._last_progress_step = 0
        self._progress_total: int | None = None
        self._latest_loss: float | None = None
        self._latest_entropy: float | None = None
        self._last_chart_step_rendered = 0

    @property
    def encoding(self) -> str:
        return "utf-8"

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    def write(self, text: str) -> int:
        if not text:
            return 0

        self._full_buffer.write(text)
        clean_text = self._ANSI_RE.sub("", text)
        for char in clean_text:
            if char == "\r":
                self._current_line = ""
            elif char == "\n":
                line = self._current_line.strip()
                if line:
                    self._history_lines.append(line)
                    self._history_lines = self._history_lines[-self._max_lines :]
                    self._capture_progress(line)
                self._current_line = ""
            elif char == "\b":
                self._current_line = self._current_line[:-1]
            else:
                self._current_line += char

        self._render()
        return len(text)

    def flush(self) -> None:
        self._render(force=True)

    def getvalue(self) -> str:
        return self._full_buffer.getvalue()

    def record_progress(self, update: dict[str, Any]) -> None:
        try:
            current = int(update.get("iteration") or 0)
            total = int(update.get("total") or 0)
        except (TypeError, ValueError):
            return
        if current <= self._last_progress_step:
            return

        loss = update.get("loss")
        entropy = update.get("entropy")
        self._last_progress_step = current
        self._progress_total = total or None
        self._latest_loss = None if loss is None or not np.isfinite(loss) else float(loss)
        self._latest_entropy = None if entropy is None or not np.isfinite(entropy) else float(entropy)
        self._loss_points.append((current, self._latest_loss))
        if self._latest_entropy is not None:
            self._progress_points.append((current, self._latest_entropy))
        self._render()

    def _render(self, *, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self._last_render < self._update_interval:
            return

        current = self._current_line.strip()
        if current:
            self._capture_progress(current)

        lines = self._history_lines[-(self._max_lines - 1) :]
        if current:
            lines = lines + [current]
        text = "\n".join(lines) if lines else "Waiting for HIPPS-DIMES output..."
        if self._placeholder is not None:
            self._placeholder.code(text)

        if self._progress_bar_placeholder is not None and self._progress_total and self._last_progress_step:
            fraction = min(max(self._last_progress_step / self._progress_total, 0.0), 1.0)
            self._progress_bar_placeholder.progress(fraction)

        if self._progress_summary_placeholder is not None and self._progress_total and self._last_progress_step:
            loss_text = "nan" if self._latest_loss is None else f"{self._latest_loss:.4g}"
            entropy_text = "nan" if self._latest_entropy is None else f"{self._latest_entropy:.4g}"
            self._progress_summary_placeholder.caption(
                f"Iteration {self._last_progress_step}/{self._progress_total} | "
                f"loss {loss_text} | entropy {entropy_text}"
            )

        if (
            self._entropy_chart_placeholder is not None
            and self._progress_points
            and self._last_progress_step > self._last_chart_step_rendered
            and (
                force
                or self._progress_total == self._last_progress_step
                or now - self._last_chart_render >= self._chart_update_interval
            )
        ):
            loss_by_iteration = {iteration: loss for iteration, loss in self._loss_points}
            chart_data = pd.DataFrame(
                {
                    "iteration": [iteration for iteration, _ in self._progress_points],
                    "entropy": [entropy for _, entropy in self._progress_points],
                }
            )
            chart_data["loss"] = chart_data["iteration"].map(loss_by_iteration)

            entropy_chart = (
                alt.Chart(chart_data)
                .mark_line(color="#0f766e", strokeWidth=2.5)
                .encode(
                    x=alt.X("iteration:Q", title="Iteration"),
                    y=alt.Y(
                        "entropy:Q",
                        title="Entropy",
                        axis=alt.Axis(
                            titleColor="#0f766e",
                            labelColor="#0f766e",
                            tickColor="#0f766e",
                            domainColor="#0f766e",
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("iteration:Q", title="Iteration"),
                        alt.Tooltip("entropy:Q", title="Entropy", format=".4g"),
                    ],
                )
            )

            loss_chart = (
                alt.Chart(chart_data)
                .mark_line(color="#b45309", strokeWidth=2.5)
                .encode(
                    x=alt.X("iteration:Q", title="Iteration"),
                    y=alt.Y(
                        "loss:Q",
                        title="Loss",
                        axis=alt.Axis(
                            titleColor="#b45309",
                            labelColor="#b45309",
                            tickColor="#b45309",
                            domainColor="#b45309",
                            orient="right",
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("iteration:Q", title="Iteration"),
                        alt.Tooltip("loss:Q", title="Loss", format=".4g"),
                    ],
                )
            )

            chart = (
                alt.layer(entropy_chart, loss_chart)
                .resolve_scale(y="independent")
                .properties(height=280)
                .configure_view(strokeOpacity=0)
            )

            self._entropy_chart_placeholder.altair_chart(
                chart,
                use_container_width=True,
            )
            self._last_chart_step_rendered = self._last_progress_step
            self._last_chart_render = now

        self._last_render = now

    def _capture_progress(self, line: str) -> None:
        match = self._PROGRESS_RE.search(line)
        if not match:
            return

        current = int(match.group("current"))
        total = int(match.group("total"))
        if current <= self._last_progress_step:
            return

        loss = float(match.group("loss"))
        entropy = float(match.group("entropy"))
        self._last_progress_step = current
        self._progress_total = total
        self._latest_loss = loss
        self._latest_entropy = entropy
        self._loss_points.append((current, loss))
        self._progress_points.append((current, entropy))


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
        a2xyz_sample,
        cmap2dmap,
        cmap2dmap_missing_data,
        compute_acf_general_theory,
        compute_m1_i,
        compute_modulus,
        compute_monomer_modulus,
        get_gpu_name,
        is_gpu_available,
        neighbor_balance_symmetric,
        run_optimization,
    )
    from hipps_dimes.numerics import cooler, hicstraw

    return HippsBindings(
        run_optimization=run_optimization,
        a2xyz_sample=a2xyz_sample,
        cmap2dmap=cmap2dmap,
        cmap2dmap_missing_data=cmap2dmap_missing_data,
        compute_m1_i=compute_m1_i,
        compute_acf_general_theory=compute_acf_general_theory,
        compute_modulus=compute_modulus,
        compute_monomer_modulus=compute_monomer_modulus,
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


def _split_input_path_group(raw_path: str) -> tuple[str, str | None]:
    base_path = _split_browser_input_path(raw_path)
    if "::" not in raw_path:
        return base_path, None
    _, group = raw_path.split("::", 1)
    stripped_group = group.strip()
    return base_path, stripped_group or None


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
    if "browser_cooler_group" not in st.session_state:
        st.session_state["browser_cooler_group"] = ""


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


def _format_cooler_group_option(group: str) -> str:
    if not group:
        return "Select a cooler group"
    if group == "/":
        return "root (no suffix)"
    if group.startswith("/resolutions/"):
        return f"{group.rsplit('/', 1)[-1]} ({group})"
    return group


@st.cache_data(show_spinner=False)
def _list_cooler_groups(input_path: str) -> tuple[str, ...]:
    _ensure_hipps_dimes_importable()
    from hipps_dimes.numerics import cooler as cooler_module

    if cooler_module is None or not hasattr(cooler_module, "fileops"):
        raise ImportError("cooler support is unavailable in this environment.")

    groups = list(cooler_module.fileops.list_coolers(input_path))
    if Path(input_path).suffix.lower() == ".cool" and "/" not in groups:
        groups.insert(0, "/")
    return tuple(dict.fromkeys(groups))


def _apply_selected_cooler_group() -> None:
    raw_input_path = st.session_state.get("input_path", "").strip()
    base_path, _ = _split_input_path_group(raw_input_path)
    if not base_path:
        _set_browser_notice("info", "Select a cooler file first.")
        return

    selected_group = st.session_state.get("browser_cooler_group", "").strip()
    normalized_base = _normalize_input_path(base_path)
    if not selected_group:
        st.session_state["input_path"] = normalized_base
        _set_browser_notice("info", "Cleared the cooler group suffix.")
        return
    if selected_group == "/":
        st.session_state["input_path"] = normalized_base
        _set_browser_notice("success", "Using the root cooler group.")
        return
    st.session_state["input_path"] = f"{normalized_base}::{selected_group}"
    _set_browser_notice("success", f"Appended cooler group `{selected_group}`")


def _render_cooler_group_picker() -> None:
    raw_input_path = st.session_state.get("input_path", "").strip()
    base_path, current_group = _split_input_path_group(raw_input_path)
    if not base_path:
        return

    suffix = Path(base_path).suffix.lower()
    if suffix not in COOLER_FILE_SUFFIXES:
        return

    normalized_base = _normalize_input_path(base_path)
    if not normalized_base:
        return

    st.markdown("")
    st.caption("Cooler groups / resolutions")
    try:
        groups = list(_list_cooler_groups(normalized_base))
    except Exception as exc:
        st.info(f"Could not inspect cooler groups for `{Path(base_path).name}`: {exc}")
        return

    if not groups:
        st.info("No cooler groups were found in this file.")
        return

    options = [""] + groups
    desired_group = current_group if current_group in groups else ""
    st.session_state["browser_cooler_group"] = desired_group
    st.selectbox(
        "Available groups",
        options=options,
        key="browser_cooler_group",
        format_func=_format_cooler_group_option,
        help="Selecting a group appends it to the Input file path automatically.",
        on_change=_apply_selected_cooler_group,
    )


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

    st.caption("Pick a local file to populate the input path. For cooler files, you can inspect and append groups below.")

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

    _render_cooler_group_picker()

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


def _normalize_existing_result_prefix(raw_value: str) -> str:
    stripped = raw_value.strip()
    if not stripped:
        return ""

    path = Path(stripped).expanduser().resolve(strict=False)
    normalized = str(path)
    for suffix in sorted(RESULT_FILE_SUFFIXES, key=len, reverse=True):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]

    checkpoint_match = re.search(r"_connectivity_matrix_iter\d+\.txt$", normalized)
    if checkpoint_match:
        return normalized[: checkpoint_match.start()]

    return normalized


def _is_pickle_result_path(raw_value: str) -> bool:
    return Path(raw_value.strip()).suffix.lower() in RESULT_PICKLE_SUFFIXES


def _empty_iteration_series_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=ITERATION_SERIES_COLUMNS)


def _parse_saved_run_parameter(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bool, int, float, list, dict)):
        return value

    text = str(value).strip()
    if text == "":
        return ""
    if text == "None":
        return None
    if text in {"True", "False"}:
        return text == "True"

    if text.startswith(("[", "{", '"')):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    try:
        if any(char in text for char in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _load_saved_run_parameters(run_parameters_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(run_parameters_path)
    if {"parameter", "value"} - set(frame.columns):
        raise ValueError("Run parameters file must contain 'parameter' and 'value' columns.")

    parameter_map = {
        str(row["parameter"]): _parse_saved_run_parameter(row["value"])
        for _, row in frame.iterrows()
    }
    return frame, parameter_map


def _bool_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def _int_value(value: Any, default: int | None = None) -> int | None:
    if value in {None, ""}:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_value(value: Any, default: float) -> float:
    if value in {None, ""}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_loaded_config(
    result_prefix: str,
    parameter_map: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parameter_map = parameter_map or {}
    overrides = overrides or {}
    input_source = parameter_map.get("input_source")
    input_path = input_source if isinstance(input_source, str) and input_source != "NumPy array" else ""

    config = {
        "mode": "load",
        "input_path": _normalize_input_path(str(input_path)) if input_path else "",
        "output_prefix": result_prefix,
        "input_type": str(parameter_map.get("input_type") or "dmap"),
        "input_format": str(parameter_map.get("input_format") or "text"),
        "selection": str(parameter_map.get("selection") or ""),
        "method": str(parameter_map.get("method") or "IS"),
        "ensemble": _int_value(parameter_map.get("ensemble"), 100),
        "iteration": _int_value(parameter_map.get("iteration"), 0),
        "alpha": _float_value(parameter_map.get("alpha"), 4.0),
        "learning_rate": _float_value(parameter_map.get("learning_rate"), 10.0),
        "momentum": _float_value(parameter_map.get("momentum"), 0.0),
        "nesterov": _bool_value(parameter_map.get("nesterov"), False),
        "use_gpu": _bool_value(parameter_map.get("use_gpu_requested"), False),
        "gpu_float32": _bool_value(parameter_map.get("gpu_float32"), False),
        "eigh_threads": _int_value(parameter_map.get("eigh_threads"), None),
        "lamd": _float_value(parameter_map.get("lamd"), 0.0),
        "reg": str(parameter_map.get("reg") or "L2"),
        "gaussian_noise_variance": _float_value(parameter_map.get("gaussian_noise_variance"), 0.0),
        "binsize": _int_value(parameter_map.get("binsize"), 25000),
        "hic_norm": str(parameter_map.get("hic_norm") or "KR"),
        "hic_unit": str(parameter_map.get("hic_unit") or "BP"),
        "balance": _bool_value(parameter_map.get("balance"), False),
        "neighbor_balance": _bool_value(parameter_map.get("neighbor_balance"), False),
        "not_normalize": _bool_value(parameter_map.get("not_normalize"), False),
        "save_steps": parameter_map.get("save_steps") or [],
        "no_log": _bool_value(parameter_map.get("no_log"), False),
        "no_xyzs": _bool_value(parameter_map.get("no_xyzs"), False),
        "ignore_missing_data": _bool_value(parameter_map.get("ignore_missing_data"), False),
        "enforce_nonnegative_connectivity_matrix": _bool_value(
            parameter_map.get("enforce_nonnegative_connectivity_matrix"),
            False,
        ),
    }

    if overrides.get("enabled"):
        override_input_path = overrides.get("input_path", "").strip()
        override_selection = overrides.get("selection", "")
        config.update(
            {
                "input_path": _normalize_input_path(override_input_path) if override_input_path else config["input_path"],
                "input_type": overrides.get("input_type", config["input_type"]),
                "input_format": overrides.get("input_format", config["input_format"]),
                "selection": override_selection if override_selection else config["selection"],
                "alpha": float(overrides.get("alpha", config["alpha"])),
                "balance": bool(overrides.get("balance", False)),
                "neighbor_balance": bool(overrides.get("neighbor_balance", False)),
                "not_normalize": bool(overrides.get("not_normalize", False)),
                "ignore_missing_data": bool(overrides.get("ignore_missing_data", False)),
                "binsize": int(overrides.get("binsize", config["binsize"])),
                "hic_norm": overrides.get("hic_norm", config["hic_norm"]),
                "hic_unit": overrides.get("hic_unit", config["hic_unit"]),
            }
        )

    return config


def _build_minimal_run_parameters_frame(config: dict[str, Any]) -> pd.DataFrame:
    rows = [
        ("input_source", config.get("input_path", "")),
        ("output_prefix", config.get("output_prefix")),
        ("input_type", config.get("input_type")),
        ("input_format", config.get("input_format")),
        ("selection", config.get("selection")),
    ]
    return pd.DataFrame(rows, columns=["parameter", "value"])


def _coerce_iteration_series_frame(data: Any) -> pd.DataFrame:
    if data is None:
        return _empty_iteration_series_frame()
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        frame = pd.DataFrame(data)
    if frame.empty:
        return _empty_iteration_series_frame()
    missing = [column for column in ITERATION_SERIES_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "Iteration series must contain columns: "
            + ", ".join(ITERATION_SERIES_COLUMNS)
        )
    return frame


def _coerce_run_parameters_artifact(data: Any) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    if data is None:
        return None, {}
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, dict):
        frame = pd.DataFrame(
            {
                "parameter": [str(key) for key in data.keys()],
                "value": list(data.values()),
            }
        )
    else:
        raise ValueError("Run parameters must be a DataFrame or dict.")

    if {"parameter", "value"} - set(frame.columns):
        raise ValueError("Run parameters must contain 'parameter' and 'value' columns.")

    parameter_map = {
        str(row["parameter"]): _parse_saved_run_parameter(row["value"])
        for _, row in frame.iterrows()
    }
    return frame, parameter_map


def _coerce_checkpoint_artifact(data: Any) -> dict[int, np.ndarray]:
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("connectivity_matrix_at_steps must be a dict.")

    checkpoints: dict[int, np.ndarray] = {}
    for step, matrix in data.items():
        checkpoints[int(step)] = np.asarray(matrix, dtype=float)
    return dict(sorted(checkpoints.items()))


def _coerce_results_mapping(results: dict[str, Any]) -> dict[str, Any]:
    if "connectivity_matrix" not in results or "dmap_final" not in results:
        raise ValueError("Pickled HIPPS-DIMES results must include 'connectivity_matrix' and 'dmap_final'.")

    normalized = dict(results)
    iteration_series = _coerce_iteration_series_frame(
        normalized.get("iteration_series", normalized.get("log"))
    )
    normalized["iteration_series"] = iteration_series
    normalized["log"] = iteration_series
    normalized["connectivity_matrix"] = np.asarray(normalized["connectivity_matrix"], dtype=float)
    normalized["dmap_final"] = np.asarray(normalized["dmap_final"], dtype=float)

    if "cmap_final" in normalized and normalized["cmap_final"] is not None:
        normalized["cmap_final"] = np.asarray(normalized["cmap_final"], dtype=float)
    if "dmap_target" in normalized and normalized["dmap_target"] is not None:
        normalized["dmap_target"] = np.asarray(normalized["dmap_target"], dtype=float)
    if "cmap_target" in normalized and normalized["cmap_target"] is not None:
        normalized["cmap_target"] = np.asarray(normalized["cmap_target"], dtype=float)
    if "xyzs" in normalized and normalized["xyzs"] is not None:
        normalized["xyzs"] = np.asarray(normalized["xyzs"], dtype=float)
    if "connectivity_matrix_at_steps" in normalized:
        normalized["connectivity_matrix_at_steps"] = _coerce_checkpoint_artifact(
            normalized["connectivity_matrix_at_steps"]
        )

    return normalized


def _load_xyz_ensemble(xyz_path: Path) -> np.ndarray:
    snapshots: list[np.ndarray] = []
    lines = xyz_path.read_text().splitlines()
    cursor = 0

    while cursor < len(lines):
        line = lines[cursor].strip()
        if not line:
            cursor += 1
            continue

        natoms = int(line)
        cursor += 1
        if cursor < len(lines) and not lines[cursor].strip():
            cursor += 1

        coords = []
        for _ in range(natoms):
            if cursor >= len(lines):
                raise ValueError(f"Unexpected end of XYZ file in {xyz_path}.")
            parts = lines[cursor].split()
            if len(parts) < 4:
                raise ValueError(f"Malformed XYZ line in {xyz_path}: {lines[cursor]!r}")
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            cursor += 1

        snapshots.append(np.asarray(coords, dtype=float))

    if not snapshots:
        raise ValueError(f"No XYZ snapshots found in {xyz_path}.")
    return np.asarray(snapshots, dtype=float)


def _load_connectivity_checkpoints(result_prefix: str) -> dict[int, np.ndarray]:
    prefix_path = Path(result_prefix)
    pattern = f"{prefix_path.name}_connectivity_matrix_iter*.txt"
    checkpoints: dict[int, np.ndarray] = {}
    for checkpoint_path in prefix_path.parent.glob(pattern):
        match = re.search(r"_connectivity_matrix_iter(\d+)\.txt$", checkpoint_path.name)
        if not match:
            continue
        checkpoints[int(match.group(1))] = np.asarray(np.loadtxt(checkpoint_path), dtype=float)
    return dict(sorted(checkpoints.items()))


def _build_loaded_target_matrices(
    bindings: HippsBindings,
    result_prefix: str,
    config: dict[str, Any],
) -> dict[str, np.ndarray | str]:
    cmap_target_path = Path(f"{result_prefix}_cmap_target.txt")
    if cmap_target_path.exists():
        raw_cmap_target = np.asarray(np.loadtxt(cmap_target_path), dtype=float)
        updates: dict[str, np.ndarray | str] = {
            "cmap_target": _normalize_contact_map(raw_cmap_target),
        }
        if config.get("input_type") == "cmap":
            if config.get("ignore_missing_data"):
                dmap_target = bindings.cmap2dmap_missing_data(
                    raw_cmap_target,
                    config["alpha"],
                    config["not_normalize"],
                )
            else:
                dmap_target = bindings.cmap2dmap(
                    raw_cmap_target,
                    config["alpha"],
                    config["not_normalize"],
                )
            updates["dmap_target"] = np.asarray(dmap_target, dtype=float)
        return updates

    if not config.get("input_path"):
        return {}

    input_matrix = _load_input_matrix(bindings, config)
    return _build_target_matrices(bindings, config, input_matrix)


def _load_pickled_results(
    bindings: HippsBindings,
    pickle_path: Path,
    request: dict[str, Any],
) -> RunArtifacts:
    with pickle_path.open("rb") as handle:
        payload = pickle.load(handle)

    if isinstance(payload, RunArtifacts):
        raw_results = payload.results
        raw_config = payload.config
        runtime_seconds = payload.runtime_seconds
        captured_stdout = payload.captured_stdout
    elif isinstance(payload, dict) and isinstance(payload.get("results"), dict):
        raw_results = payload["results"]
        raw_config = payload.get("config", {})
        runtime_seconds = float(payload.get("runtime_seconds", np.nan))
        captured_stdout = str(payload.get("captured_stdout", ""))
    elif hasattr(payload, "results") and isinstance(getattr(payload, "results"), dict):
        raw_results = getattr(payload, "results")
        raw_config = getattr(payload, "config", {})
        runtime_seconds = float(getattr(payload, "runtime_seconds", np.nan))
        captured_stdout = str(getattr(payload, "captured_stdout", ""))
    elif isinstance(payload, dict):
        raw_results = payload
        raw_config = {}
        runtime_seconds = np.nan
        captured_stdout = ""
    else:
        raise ValueError(
            "Unsupported pickle payload. Expected a HIPPS-DIMES results dict or a RunArtifacts-like object."
        )

    results = _coerce_results_mapping(raw_results)
    run_parameters_frame, parameter_map = _coerce_run_parameters_artifact(results.get("run_parameters"))

    pickle_prefix = str(pickle_path.with_suffix(""))
    config = _build_loaded_config(pickle_prefix, parameter_map, request.get("overrides"))
    if isinstance(raw_config, dict):
        config.update(raw_config)
    config["mode"] = "load"
    config["loaded_from"] = str(pickle_path)
    config["output_prefix"] = str(config.get("output_prefix") or pickle_prefix)
    if config.get("input_path"):
        config["input_path"] = _normalize_input_path(str(config["input_path"]))

    if run_parameters_frame is None:
        results["run_parameters"] = _build_minimal_run_parameters_frame(config)
    else:
        results["run_parameters"] = run_parameters_frame

    if "matrix_target_error" not in results and "dmap_target" not in results and "cmap_target" not in results:
        try:
            target_prefix = str(config.get("output_prefix") or pickle_prefix)
            results.update(_build_loaded_target_matrices(bindings, target_prefix, config))
        except Exception as exc:
            results["matrix_target_error"] = str(exc)

    return RunArtifacts(
        results=results,
        runtime_seconds=runtime_seconds if np.isfinite(runtime_seconds) else np.nan,
        captured_stdout=captured_stdout,
        config=config,
    )


def _load_existing_results(bindings: HippsBindings, request: dict[str, Any]) -> RunArtifacts:
    raw_result_reference = request["result_prefix"].strip()
    if _is_pickle_result_path(raw_result_reference):
        pickle_path = Path(raw_result_reference).expanduser().resolve(strict=False)
        if not pickle_path.exists():
            raise FileNotFoundError(f"Missing pickle file: {pickle_path}")
        return _load_pickled_results(bindings, pickle_path, request)

    result_prefix = _normalize_existing_result_prefix(request["result_prefix"])
    if not result_prefix:
        raise ValueError("A result prefix or an existing HIPPS-DIMES output file is required.")

    connectivity_path = Path(f"{result_prefix}_connectivity_matrix.txt")
    dmap_path = Path(f"{result_prefix}_dmap_final.txt")
    if not connectivity_path.exists():
        raise FileNotFoundError(f"Missing required file: {connectivity_path}")
    if not dmap_path.exists():
        raise FileNotFoundError(f"Missing required file: {dmap_path}")

    run_parameters_path = Path(f"{result_prefix}_run_parameters.csv")
    if run_parameters_path.exists():
        run_parameters, parameter_map = _load_saved_run_parameters(run_parameters_path)
    else:
        run_parameters = None
        parameter_map = {}

    config = _build_loaded_config(result_prefix, parameter_map, request.get("overrides"))

    results: dict[str, Any] = {
        "iteration_series": _empty_iteration_series_frame(),
        "log": _empty_iteration_series_frame(),
        "run_parameters": run_parameters if run_parameters is not None else _build_minimal_run_parameters_frame(config),
        "dmap_final": np.asarray(np.loadtxt(dmap_path), dtype=float),
        "connectivity_matrix": np.asarray(np.loadtxt(connectivity_path), dtype=float),
    }

    iteration_series_path = Path(f"{result_prefix}_iteration_series.csv")
    if iteration_series_path.exists():
        iteration_series = pd.read_csv(iteration_series_path)
        results["iteration_series"] = iteration_series
        results["log"] = iteration_series

    cmap_final_path = Path(f"{result_prefix}_cmap_final.txt")
    if cmap_final_path.exists():
        results["cmap_final"] = np.asarray(np.loadtxt(cmap_final_path), dtype=float)

    checkpoints = _load_connectivity_checkpoints(result_prefix)
    if checkpoints:
        results["connectivity_matrix_at_steps"] = checkpoints

    xyz_path = Path(f"{result_prefix}.xyz")
    if xyz_path.exists():
        try:
            results["xyzs"] = _load_xyz_ensemble(xyz_path)
        except Exception as exc:
            results["xyz_load_error"] = str(exc)

    try:
        results.update(_build_loaded_target_matrices(bindings, result_prefix, config))
    except Exception as exc:
        results["matrix_target_error"] = str(exc)

    return RunArtifacts(
        results=results,
        runtime_seconds=np.nan,
        captured_stdout="",
        config=config,
    )


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


def _load_input_matrix(bindings: HippsBindings, config: dict[str, Any]) -> np.ndarray:
    input_type = config["input_type"]
    input_path = config["input_path"]
    input_format = config["input_format"]

    if input_type == "cmap":
        return np.asarray(_load_contact_map_from_source(bindings, config))
    if input_format == "text":
        return np.asarray(np.loadtxt(input_path))
    if input_format == "npy":
        return np.asarray(np.load(input_path))
    raise ValueError(
        f"input_type='{input_type}' only supports input_format={SUPPORTED_INPUT_FORMATS[input_type]}"
    )


def _normalize_contact_map(cmap: np.ndarray) -> np.ndarray:
    cmap = np.asarray(cmap, dtype=float)
    max_value = np.nanmax(cmap)
    if not np.isfinite(max_value) or max_value == 0.0:
        return cmap
    return cmap / max_value


def _build_target_matrices(
    bindings: HippsBindings,
    config: dict[str, Any],
    input_matrix: np.ndarray,
) -> dict[str, np.ndarray]:
    input_type = config["input_type"]

    if input_type == "dmap":
        return {"dmap_target": np.asarray(input_matrix, dtype=float)}

    if input_type == "ddmap":
        dmap_target = np.sqrt((8.0 / (3.0 * np.pi)) * np.asarray(input_matrix, dtype=float))
        return {"dmap_target": dmap_target}

    if input_type == "cmap":
        cmap_target = np.asarray(input_matrix, dtype=float)
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
                save_steps = st.text_input(
                    "Save steps",
                    value="",
                    help="Comma-separated iterations. When Gaussian noise is enabled, HIPPS-DIMES requires an output prefix.",
                )
                no_log = st.checkbox("Skip CSV logs", value=False)
                no_xyzs = st.checkbox("Skip XYZ generation", value=False)
                ignore_missing_data = st.checkbox("Ignore missing data", value=False)
                enforce_nonnegative = st.checkbox("Enforce nonnegative springs", value=False)

            run_clicked = st.form_submit_button("Run HIPPS-DIMES", use_container_width=True)

        if not run_clicked:
            return None

        config = {
            "mode": "run",
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


def _render_load_results_sidebar() -> dict[str, Any] | None:
    with st.sidebar:
        st.header("Load Results")
        st.caption(
            "Load an existing HIPPS-DIMES result set from its output prefix, any one of its standard output files, "
            "or a trusted Python pickle file."
        )
        st.warning("Only load `.pkl` or `.pickle` files you trust.")

        with st.form("load_results_form", clear_on_submit=False):
            result_prefix = st.text_input(
                "Result prefix or output file",
                value="",
                help=(
                    "Examples: `/path/to/run`, `/path/to/run_connectivity_matrix.txt`, "
                    "`/path/to/run_iteration_series.csv`, or `/path/to/run_results.pkl`."
                ),
            )

            override_enabled = st.checkbox(
                "Override input metadata for target reconstruction",
                value=False,
                help="Use this if `run_parameters.csv` is missing or the original input file moved.",
            )
            with st.expander("Optional input metadata override", expanded=override_enabled):
                override_input_path = st.text_input(
                    "Original input path",
                    value="",
                    help="Only needed if you want the app to rebuild target matrices from the original input.",
                )
                input_col, format_col = st.columns(2)
                override_input_type = input_col.selectbox("Input type", ["cmap", "dmap", "ddmap"], index=0, key="load_input_type")
                override_input_format = format_col.selectbox("Input format", ["text", "npy", "cooler", "hic"], index=0, key="load_input_format")
                override_selection = st.text_input("Selection / region", value="", key="load_selection")
                override_alpha = st.number_input("Alpha", min_value=0.1, value=4.0, step=0.1, key="load_alpha")
                binsize_col, norm_col = st.columns(2)
                override_binsize = binsize_col.number_input("Hi-C binsize", min_value=1, value=25000, step=1000, key="load_binsize")
                override_hic_norm = norm_col.selectbox("Hi-C norm", ["KR", "VC", "NONE"], index=0, key="load_hic_norm")
                unit_col, balance_col = st.columns(2)
                override_hic_unit = unit_col.selectbox("Hi-C unit", ["BP", "FRAG"], index=0, key="load_hic_unit")
                override_balance = balance_col.checkbox("Cooler balance", value=False, key="load_balance")
                override_neighbor_balance = st.checkbox("Neighbor balance", value=False, key="load_neighbor_balance")
                override_not_normalize = st.checkbox("Skip contact-map normalization", value=False, key="load_not_normalize")
                override_ignore_missing_data = st.checkbox("Ignore missing data", value=False, key="load_ignore_missing_data")

            load_clicked = st.form_submit_button("Load existing results", use_container_width=True)

        if not load_clicked:
            return None

        return {
            "mode": "load",
            "result_prefix": result_prefix,
            "overrides": {
                "enabled": bool(override_enabled),
                "input_path": override_input_path,
                "input_type": override_input_type,
                "input_format": override_input_format,
                "selection": override_selection,
                "alpha": float(override_alpha),
                "binsize": int(override_binsize),
                "hic_norm": override_hic_norm,
                "hic_unit": override_hic_unit,
                "balance": bool(override_balance),
                "neighbor_balance": bool(override_neighbor_balance),
                "not_normalize": bool(override_not_normalize),
                "ignore_missing_data": bool(override_ignore_missing_data),
            },
        }


def _validate_config(config: dict[str, Any]) -> tuple[bool, str | None]:
    if not config["input_path"].strip():
        return False, "An input file path is required."
    supported_formats = SUPPORTED_INPUT_FORMATS[config["input_type"]]
    if config["input_format"] not in supported_formats:
        return False, (
            f"Input type '{config['input_type']}' only supports formats: "
            f"{', '.join(supported_formats)}."
        )
    if config["input_format"] in {"cooler", "hic"} and not config["selection"].strip():
        return False, "Selection is required for cooler and .hic inputs."
    try:
        save_steps = _parse_save_steps(config["save_steps"])
    except ValueError:
        return False, "Save steps must be a comma-separated list of integers."
    if save_steps is not None and config["gaussian_noise_variance"] > 0.0 and not config["output_prefix"].strip():
        return False, "Output prefix is required when save steps are used with Gaussian noise."
    if config["gaussian_noise_variance"] > 0.0 and config["lamd"] > 0.0:
        return False, "Gaussian noise variance cannot be combined with lambda regularization."
    return True, None


def _validate_load_request(request: dict[str, Any]) -> tuple[bool, str | None]:
    if not request["result_prefix"].strip():
        return False, "A result prefix or existing HIPPS-DIMES output file path is required."
    return True, None


def _run_model(
    bindings: HippsBindings,
    config: dict[str, Any],
    live_output_placeholder: Any | None = None,
    live_progress_bar_placeholder: Any | None = None,
    live_progress_summary_placeholder: Any | None = None,
    live_entropy_chart_placeholder: Any | None = None,
) -> RunArtifacts:
    normalized_path = _normalize_input_path(config["input_path"])
    output_prefix = _make_output_prefix(config["output_prefix"])
    save_steps = _parse_save_steps(config["save_steps"])
    execution_config = dict(config)
    execution_config["input_path"] = normalized_path
    execution_config["output_prefix"] = output_prefix
    execution_config["save_steps"] = save_steps
    input_matrix = _load_input_matrix(bindings, execution_config)

    kwargs = {
        "input_path": normalized_path,
        "input_matrix": input_matrix,
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
        "verbose": True,
    }

    stdout_buffer = _StreamlitOutputBuffer(
        live_output_placeholder,
        progress_bar_placeholder=live_progress_bar_placeholder,
        progress_summary_placeholder=live_progress_summary_placeholder,
        entropy_chart_placeholder=live_entropy_chart_placeholder,
    )
    run_signature = inspect.signature(bindings.run_optimization)
    native_progress_supported = (
        "progress_callback" in run_signature.parameters
        and "show_progress" in run_signature.parameters
    )
    if native_progress_supported:
        kwargs["progress_callback"] = stdout_buffer.record_progress
        kwargs["show_progress"] = False
        kwargs["verbose"] = False
    start = time.perf_counter()
    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stdout_buffer):
            results = bindings.run_optimization(**kwargs)
    finally:
        stdout_buffer.flush()
    runtime_seconds = time.perf_counter() - start

    try:
        results.update(_build_target_matrices(bindings, execution_config, input_matrix))
    except Exception as exc:
        results["matrix_target_error"] = str(exc)

    return RunArtifacts(
        results=results,
        runtime_seconds=runtime_seconds,
        captured_stdout=stdout_buffer.getvalue().strip(),
        config=execution_config,
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
    corner_labels: tuple[tuple[str, str], ...] = (),
) -> go.Figure:
    plot_matrix = np.asarray(matrix)
    colorbar_title = "value"
    if log_transform:
        plot_matrix = np.where(plot_matrix > 0, np.log(plot_matrix), np.nan)
        colorbar_title = "log(value)"
    n_rows, n_cols = plot_matrix.shape
    row_index, col_index = np.indices((n_rows, n_cols))
    customdata = np.dstack((row_index, col_index))
    x_coords = np.arange(n_cols, dtype=float) + 0.5
    y_coords = np.arange(n_rows, dtype=float) + 0.5

    figure = go.Figure(
        data=[
            go.Heatmap(
                x=x_coords,
                y=y_coords,
                z=plot_matrix,
                customdata=customdata,
                colorscale=colorscale,
                zmid=midpoint,
                colorbar=dict(title=colorbar_title),
                hovertemplate="row=%{customdata[0]}<br>col=%{customdata[1]}<br>value=%{z:.4g}<extra></extra>",
            )
        ]
    )
    figure.update_layout(
        title=title,
        height=620,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="Locus",
        yaxis_title="Locus",
        dragmode="pan",
    )
    for text, corner in corner_labels:
        x = 0.03 if "left" in corner else 0.97
        y = 0.03 if "lower" in corner else 0.97
        xanchor = "left" if "left" in corner else "right"
        yanchor = "bottom" if "lower" in corner else "top"
        figure.add_annotation(
            x=x,
            y=y,
            xref="x domain",
            yref="y domain",
            text=text,
            showarrow=False,
            xanchor=xanchor,
            yanchor=yanchor,
            font=dict(size=13, color="#182026"),
            bgcolor="rgba(255, 250, 243, 0.88)",
            bordercolor="rgba(24, 32, 38, 0.18)",
            borderwidth=1,
            borderpad=4,
        )
    figure.update_xaxes(range=[0, n_cols], constrain="domain")
    figure.update_yaxes(range=[n_rows, 0], scaleanchor="x", scaleratio=1)
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


def _plot_dual_axis_series(
    x1: np.ndarray,
    y1: np.ndarray,
    label1: str,
    x2: np.ndarray,
    y2: np.ndarray,
    label2: str,
    log_x: bool = True,
    log_y1: bool = False,
    log_y2: bool = True,
) -> go.Figure:
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=x1,
            y=y1,
            mode="lines",
            line=dict(color="#d97706", width=2.5),
            name=label1,
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=x2,
            y=y2,
            mode="lines",
            line=dict(color="#0f766e", width=2.5),
            name=label2,
        ),
        secondary_y=True,
    )
    figure.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20))
    figure.update_xaxes(type="log" if log_x else "linear")
    figure.update_yaxes(type="log" if log_y1 else "linear", secondary_y=False)
    figure.update_yaxes(type="log" if log_y2 else "linear", secondary_y=True)
    return figure


def _render_overview(artifacts: RunArtifacts) -> None:
    results = artifacts.results
    iteration_series = results["iteration_series"]
    connectivity_matrix = results["connectivity_matrix"]
    run_parameters = results["run_parameters"]
    final_loss = iteration_series["loss"].iloc[-1] if not iteration_series.empty else np.nan
    final_entropy = iteration_series["entropy"].iloc[-1] if not iteration_series.empty else np.nan
    runtime_display = f"{artifacts.runtime_seconds:.2f}s" if np.isfinite(artifacts.runtime_seconds) else "N/A"
    runtime_note = (
        "Wall-clock time inside the Streamlit run."
        if np.isfinite(artifacts.runtime_seconds)
        else "Unavailable when loading results from disk."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _metric_card("Matrix size", str(connectivity_matrix.shape[0]), "Loci in the final connectivity matrix.")
    with col2:
        _metric_card("Runtime", runtime_display, runtime_note)
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
            f"Available XYZs: `{'xyzs' in results}`",
            f"Final contact map: `{'cmap_final' in results}`",
            f"Intermediate checkpoints: `{len(results.get('connectivity_matrix_at_steps', {}))}`",
        ]
        if artifacts.config.get("loaded_from"):
            lines.append(f"Loaded from: `{artifacts.config['loaded_from']}`")
        st.markdown("\n".join(f"- {line}" for line in lines))

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
        "Final distance map": (distance_matrix, "Viridis", None, False, distance_title),
        "Connectivity matrix": (results["connectivity_matrix"], "Tealrose", 0.0, False, "Connectivity matrix"),
    }
    if "cmap_final" in results and "cmap_target" in results:
        options["Final contact map"] = (
            _combine_triangle_matrices(results["cmap_target"], results["cmap_final"]),
            "Oranges",
            None,
            True,
            "Contact map comparison",
        )
    elif "cmap_final" in results:
        options["Final contact map"] = (results["cmap_final"], "Oranges", None, True, "Final contact map")

    labels = list(options.keys())
    selected = st.selectbox("Matrix", labels, index=0)
    matrix, colorscale, midpoint, log_transform, title = options[selected]
    corner_labels: tuple[tuple[str, str], ...] = ()
    if selected in {"Final distance map", "Final contact map"} and (
        (selected == "Final distance map" and "dmap_target" in results)
        or (selected == "Final contact map" and "cmap_target" in results)
    ):
        corner_labels = (
            ("Target", "lower left"),
            ("HIPPS-DIMES", "upper right"),
        )
    st.plotly_chart(
        _plot_matrix(matrix, title, colorscale, midpoint, log_transform=log_transform, corner_labels=corner_labels),
        use_container_width=True,
        config={"scrollZoom": True},
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
                config={"scrollZoom": True},
            )


def _render_convergence(results: dict[str, Any]) -> None:
    iteration_series = results["iteration_series"]
    if iteration_series.empty:
        st.info("No iteration-series data returned.")
        return
    st.plotly_chart(_plot_convergence(iteration_series), use_container_width=True)
    st.dataframe(iteration_series.tail(25), use_container_width=True, hide_index=True)


def _render_structures(bindings: HippsBindings, artifacts: RunArtifacts) -> None:
    results = artifacts.results
    xyzs = results.get("xyzs")
    if xyzs is None:
        if "xyz_load_error" in results:
            st.warning(f"Could not load the XYZ ensemble: {results['xyz_load_error']}")
        else:
            st.info("No XYZ ensemble is currently available for this result set.")

        default_ensemble = int(artifacts.config.get("ensemble") or 100)
        ensemble = st.number_input(
            "Ensemble size for on-demand generation",
            min_value=1,
            value=max(default_ensemble, 1),
            step=10,
            key="structure_generate_ensemble",
            help="Generate structures in the app with HIPPS-DIMES `a2xyz_sample(connectivity_matrix, ensemble=ensemble)`.",
        )
        if st.button("Generate structures in app", use_container_width=True, key="generate_xyzs_in_app"):
            with st.spinner("Generating structures from the connectivity matrix..."):
                xyzs = bindings.a2xyz_sample(
                    results["connectivity_matrix"],
                    ensemble=int(ensemble),
                )
            results["xyzs"] = xyzs
            artifacts.config["ensemble"] = int(ensemble)
            results.pop("xyz_load_error", None)

        if xyzs is None:
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
        figure = _plot_dual_axis_series(
            acf[:, 0],
            acf[:, 1],
            f"ACF ({i}, {j})",
            two_point_msd[:, 0],
            two_point_msd[:, 1],
            f"2-point MSD ({i}, {j})",
            log_x=True,
        )
        acf_t0 = float(acf[0, 1] + 0.5 * two_point_msd[0, 1])
        acf_one_over_e = acf_t0 / np.e
        figure.add_trace(
            go.Scatter(
                x=[acf[0, 0], acf[-1, 0]],
                y=[acf_one_over_e, acf_one_over_e],
                mode="lines",
                line=dict(color="#9c6644", width=2, dash="dash"),
                name="ACF(0) / e",
                hovertemplate="ACF(0)/e=%{y:.4g}<extra></extra>",
            ),
            secondary_y=False,
        )
        figure.update_xaxes(title_text="Time")
        figure.update_yaxes(title_text="ACF", secondary_y=False)
        figure.update_yaxes(title_text="2-point MSD", secondary_y=True)
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
        _render_structures(bindings, artifacts)
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

    with st.sidebar:
        st.header("Data Source")
        data_source_mode = st.radio(
            "Mode",
            ["Run HIPPS-DIMES", "Load existing results"],
            index=0,
            label_visibility="collapsed",
        )

    request = (
        _render_sidebar(bindings)
        if data_source_mode == "Run HIPPS-DIMES"
        else _render_load_results_sidebar()
    )
    if request is not None:
        if request["mode"] == "run":
            valid, error_message = _validate_config(request)
            if not valid:
                st.session_state.pop("artifacts", None)
                st.error(error_message)
            else:
                with st.status("Running HIPPS-DIMES...", expanded=True) as run_status:
                    st.caption("Live HIPPS-DIMES progress")
                    live_progress_bar_placeholder = st.empty()
                    live_progress_summary_placeholder = st.empty()
                    live_entropy_chart_placeholder = st.empty()
                    try:
                        st.session_state["artifacts"] = _run_model(
                            bindings,
                            request,
                            live_progress_bar_placeholder=live_progress_bar_placeholder,
                            live_progress_summary_placeholder=live_progress_summary_placeholder,
                            live_entropy_chart_placeholder=live_entropy_chart_placeholder,
                        )
                    except Exception as exc:
                        st.session_state.pop("artifacts", None)
                        run_status.update(label="HIPPS-DIMES run failed", state="error", expanded=True)
                        st.exception(exc)
                    else:
                        run_status.update(label="HIPPS-DIMES run complete", state="complete", expanded=False)
        else:
            valid, error_message = _validate_load_request(request)
            if not valid:
                st.session_state.pop("artifacts", None)
                st.error(error_message)
            else:
                with st.status("Loading results...", expanded=False) as load_status:
                    try:
                        st.session_state["artifacts"] = _load_existing_results(bindings, request)
                    except Exception as exc:
                        st.session_state.pop("artifacts", None)
                        load_status.update(label="Result loading failed", state="error", expanded=True)
                        st.exception(exc)
                    else:
                        load_status.update(label="Results loaded", state="complete", expanded=False)

    artifacts = st.session_state.get("artifacts")
    if artifacts is None:
        st.info(
            "Configure a run in the sidebar, then execute HIPPS-DIMES. "
            "The app will keep the most recent result in memory for interactive inspection."
        )
        return

    _render_results(bindings, artifacts)
