from __future__ import annotations

import json
import os
import queue
import re
import shutil
import html
import io
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from uuid import uuid4
import ast

import streamlit as st
from openpyxl import Workbook


CURRENT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = CURRENT_DIR / "config"
SCRIPT_PATH = CURRENT_DIR / "feature_woe_analysis.py"
WEB_RUNS_DIR = CURRENT_DIR / "feature_analysis_output" / "web_runs"
WEB_HISTORY_PATH = CONFIG_DIR / "feature_web_history.json"
WEB_RUNTIME_DIR = CURRENT_DIR / "feature_analysis_output" / "web_runtime"


def build_feature_desc_template_bytes() -> bytes:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sheet1"
    sheet["A1"] = "特征码"
    sheet["B1"] = "含义"
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def inject_ui_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --dynamic-header-offset: 72px;
        }
        .block-container {
            max-width: 1280px;
            /* 根据实际顶部栏高度动态偏移，避免内容被覆盖 */
            padding-top: calc(var(--dynamic-header-offset) + 0.8rem) !important;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            letter-spacing: 0.2px;
        }
        .block-container h1 {
            line-height: 1.22 !important;
            overflow: visible !important;
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.75rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 42px;
            border-radius: 10px 10px 0 0;
            padding: 0 14px;
        }
        .stButton>button[kind="primary"] {
            height: 46px;
            font-weight: 700;
            border-radius: 12px;
        }
        .stDownloadButton>button {
            border-radius: 10px;
        }
        .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
            border-radius: 10px;
        }
        .stFileUploader {
            border-radius: 12px;
        }
        /* 上传按钮文案：Browse files -> 选择文件 */
        [data-testid="stFileUploaderDropzone"] button[kind="secondary"] {
            color: transparent !important;
            position: relative;
        }
        [data-testid="stFileUploaderDropzone"] button[kind="secondary"]::after {
            content: "选择文件";
            color: #2d3142;
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            white-space: nowrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <script>
        (function() {
          const updateHeaderOffset = () => {
            const header = window.parent?.document?.querySelector('header[data-testid="stHeader"]')
              || document.querySelector('header[data-testid="stHeader"]');
            if (!header) return;
            const h = Math.max(0, Math.ceil(header.getBoundingClientRect().height || 0));
            document.documentElement.style.setProperty('--dynamic-header-offset', `${h}px`);
          };
          updateHeaderOffset();
          window.addEventListener('resize', updateHeaderOffset);
          const observer = new MutationObserver(updateHeaderOffset);
          observer.observe(document.body, { childList: true, subtree: true, attributes: true });
          setInterval(updateHeaderOffset, 800);
        })();
        </script>
        """,
        unsafe_allow_html=True,
    )


def list_config_files(patterns: List[str]) -> List[Path]:
    if not CONFIG_DIR.exists():
        return []
    files: List[Path] = []
    for pattern in patterns:
        files.extend(sorted(CONFIG_DIR.glob(pattern)))
    return sorted(set(files), key=lambda p: p.name.lower())


def load_sheet_toggle_options() -> dict:
    config_path = CONFIG_DIR / "feature_sheet_output_config.json"
    if not config_path.exists():
        return {}
    config_data: dict = {}
    raw_text = config_path.read_text(encoding="utf-8")
    cleaned_lines: List[str] = []
    for line in raw_text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("//"):
            comment_match = re.match(r'^\s*//\s*"([^"]+)"\s*:', line)
            if comment_match:
                config_data[comment_match.group(1)] = False
            continue
        cleaned_lines.append(line)
    try:
        loaded = json.loads("\n".join(cleaned_lines))
        if isinstance(loaded, dict):
            for key, value in loaded.items():
                config_data[str(key)] = bool(value)
    except Exception:
        pass
    return config_data


def assert_dir_writable(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    probe = directory / f".write_probe_{uuid4().hex}.tmp"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink(missing_ok=True)


def create_writable_run_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_candidates = [
        WEB_RUNTIME_DIR,
        Path(tempfile.gettempdir()) / "feature_woe_web_runtime",
        Path.home() / "feature_woe_web_runtime",
    ]
    for base_dir in base_candidates:
        try:
            assert_dir_writable(base_dir)
            run_dir = base_dir / f"run_{timestamp}_{uuid4().hex[:8]}"
            run_dir.mkdir(parents=True, exist_ok=True)
            assert_dir_writable(run_dir)
            return run_dir
        except Exception:
            continue
    raise PermissionError("无法创建可写运行目录，请检查当前用户对本地目录的写权限。")


def save_uploaded_file(
    uploaded_file,
    target_path: Path,
    fallback_dirs: Optional[List[Path]] = None,
) -> Path:
    data = bytes(uploaded_file.getbuffer())
    candidates = [target_path]
    suffix = target_path.suffix or ".bin"
    candidates.append(target_path.with_name(f"{target_path.stem}_{uuid4().hex}{suffix}"))
    for fallback_dir in (fallback_dirs or []):
        try:
            fallback_dir.mkdir(parents=True, exist_ok=True)
            candidates.append(fallback_dir / target_path.name)
            candidates.append(fallback_dir / f"{target_path.stem}_{uuid4().hex}{suffix}")
        except Exception:
            continue
    last_error: Optional[Exception] = None
    for path in candidates:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return path
        except PermissionError as exc:
            last_error = exc
            continue
    if last_error:
        raise last_error
    raise PermissionError(f"无法写入上传文件: {target_path}")


def safe_upload_filename(uploaded_name: str, fallback: str = "uploaded_file") -> str:
    raw = (uploaded_name or "").strip().replace("\\", "/")
    name = Path(raw).name.strip()
    if not name:
        return fallback
    return name


def save_uploaded_text(uploaded_file, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    content = uploaded_file.getvalue().decode("utf-8")
    json.loads(content)
    target_path.write_text(content, encoding="utf-8")
    return target_path


def write_json_config(config_obj: dict, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(config_obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target_path


def run_analysis(
    input_path: Path,
    output_base_path: Path,
    status_source: int,
    feature_prefixes: str,
    sheet_config_path: Optional[Path],
    display_config_path: Optional[Path],
    feature_desc_path: Optional[Path],
) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        "-u",
        str(SCRIPT_PATH),
        str(input_path),
        "--output-file",
        str(output_base_path),
        "--status-source",
        str(status_source),
    ]
    if feature_prefixes.strip():
        cmd.extend(["--feature-prefixes", feature_prefixes.strip()])
    if sheet_config_path:
        cmd.extend(["--sheet-config-file", str(sheet_config_path)])
    if display_config_path:
        cmd.extend(["--display-config-file", str(display_config_path)])
    if feature_desc_path:
        cmd.extend(["--feature-desc-file", str(feature_desc_path)])

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(CURRENT_DIR), env=env)


def update_progress_state(line: str, state: dict) -> int:
    text = (line or "").strip()
    if not text:
        return int(state.get("progress", 1))

    progress = int(state.get("progress", 1))
    if "[main] load_input start" in text:
        progress = max(progress, 5)
    if "[main] load_input done" in text:
        progress = max(progress, 10)
    if "[main] normalize_target done" in text:
        progress = max(progress, 15)

    if text.startswith("[segment] available="):
        match = re.search(r"\[segment\]\s+available=(.+)$", text)
        if match:
            raw = match.group(1).strip()
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, list):
                    state["total_segments"] = max(len(parsed), 1)
            except Exception:
                pass

    if text.startswith("[segment] processing="):
        state["current_segment_index"] = int(state.get("current_segment_index", 0)) + 1
        state["segment_progress"] = 0.0

    segment_inner = float(state.get("segment_progress", 0.0))
    if "[main] select_features done" in text:
        segment_inner = max(segment_inner, 0.10)
    if "[woe] start" in text:
        segment_inner = max(segment_inner, 0.15)
    if "[main] woe_done" in text:
        segment_inner = max(segment_inner, 0.60)
    if "[summary] start" in text:
        segment_inner = max(segment_inner, 0.65)
    if "[main] summary_done" in text:
        segment_inner = max(segment_inner, 0.85)
    if "[excel] start" in text:
        segment_inner = max(segment_inner, 0.88)
    if "[excel] saved" in text:
        segment_inner = max(segment_inner, 0.99)

    woe_match = re.search(r"\[woe\]\s+progress=(\d+)/(\d+)", text)
    if woe_match:
        done = int(woe_match.group(1))
        total = max(int(woe_match.group(2)), 1)
        segment_inner = max(segment_inner, 0.15 + (done / total) * (0.60 - 0.15))

    summary_match = re.search(r"\[summary\]\s+progress=(\d+)/(\d+)", text)
    if summary_match:
        done = int(summary_match.group(1))
        total = max(int(summary_match.group(2)), 1)
        segment_inner = max(segment_inner, 0.65 + (done / total) * (0.85 - 0.65))

    excel_match = re.search(r"\[excel\]\s+wrote_sheet=.*progress=(\d+)/(\d+)", text)
    if excel_match:
        done = int(excel_match.group(1))
        total = max(int(excel_match.group(2)), 1)
        segment_inner = max(segment_inner, 0.88 + (done / total) * (0.98 - 0.88))

    state["segment_progress"] = min(max(segment_inner, 0.0), 1.0)

    total_segments = int(state.get("total_segments", 1))
    current_segment_index = int(state.get("current_segment_index", 0))
    if current_segment_index > 0:
        base = 15.0
        span = 84.0
        completed_segments = max(current_segment_index - 1, 0)
        global_ratio = (completed_segments + state["segment_progress"]) / max(total_segments, 1)
        progress = max(progress, int(base + span * global_ratio))

    if "[main] total_elapsed" in text:
        progress = 100

    state["progress"] = min(max(progress, 1), 100)
    return int(state["progress"])


def run_analysis_with_live_progress(
    input_path: Path,
    output_base_path: Path,
    status_source: int,
    feature_prefixes: str,
    sheet_config_path: Optional[Path],
    display_config_path: Optional[Path],
    feature_desc_path: Optional[Path],
    progress_placeholder,
    stage_placeholder,
    log_placeholder,
) -> tuple[int, str, str]:
    def render_live_log(lines: List[str]) -> None:
        content = "\n".join(lines[-120:])
        escaped = html.escape(content)
        log_placeholder.markdown(
            (
                "<div style='height:280px; overflow-y:auto; "
                "border:1px solid #ddd; border-radius:8px; padding:10px; "
                "background:#0E1117; color:#F5F5F5; font-family:Consolas,Monaco,monospace; "
                "font-size:12px; white-space:pre-wrap;'>"
                f"{escaped}</div>"
            ),
            unsafe_allow_html=True,
        )

    cmd = [
        sys.executable,
        "-u",
        str(SCRIPT_PATH),
        str(input_path),
        "--output-file",
        str(output_base_path),
        "--status-source",
        str(status_source),
    ]
    if feature_prefixes.strip():
        cmd.extend(["--feature-prefixes", feature_prefixes.strip()])
    if sheet_config_path:
        cmd.extend(["--sheet-config-file", str(sheet_config_path)])
    if display_config_path:
        cmd.extend(["--display-config-file", str(display_config_path)])
    if feature_desc_path:
        cmd.extend(["--feature-desc-file", str(feature_desc_path)])

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        cmd,
        cwd=str(CURRENT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    progress_percent = 1
    progress_placeholder.progress(progress_percent, text="准备启动分析...")
    stdout_lines: List[str] = []
    log_queue: "queue.Queue[Optional[str]]" = queue.Queue()
    last_log_ts = time.time()
    start_ts = time.time()
    progress_state = {
        "progress": 1,
        "total_segments": 1,
        "current_segment_index": 0,
        "segment_progress": 0.0,
    }

    def _reader(pipe, q: "queue.Queue[Optional[str]]") -> None:
        try:
            if pipe is None:
                q.put(None)
                return
            for line in iter(pipe.readline, ""):
                q.put(line)
        finally:
            q.put(None)

    reader_thread = threading.Thread(target=_reader, args=(process.stdout, log_queue), daemon=True)
    reader_thread.start()

    stream_closed = False
    while True:
        got_new_line = False
        while True:
            try:
                item = log_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                stream_closed = True
                break
            stripped = item.rstrip("\n")
            stdout_lines.append(stripped)
            got_new_line = True
            last_log_ts = time.time()
            progress_percent = update_progress_state(stripped, progress_state)
            stage_placeholder.caption(f"当前阶段：{stripped}")

        elapsed = int(time.time() - start_ts)
        since_last_log = int(time.time() - last_log_ts)
        progress_placeholder.progress(
            progress_percent,
            text=f"进度 {progress_percent}% · 已运行 {elapsed}s · 最近日志 {since_last_log}s 前",
        )
        if got_new_line:
            render_live_log(stdout_lines)

        if stream_closed and process.poll() is not None:
            break
        time.sleep(0.2)

    return_code = process.wait()
    final_stdout = "\n".join(stdout_lines)
    if return_code == 0:
        progress_placeholder.progress(100, text="进度 100%")
    return return_code, final_stdout, ""


def load_web_history() -> List[dict]:
    if not WEB_HISTORY_PATH.exists():
        return []
    try:
        data = json.loads(WEB_HISTORY_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def save_web_history(history: List[dict]) -> None:
    WEB_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    WEB_HISTORY_PATH.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def delete_history_item_files(item: dict) -> None:
    run_dir_text = str(item.get("run_dir", "")).strip()
    if run_dir_text:
        run_dir = Path(run_dir_text)
        remove_tree_with_retry(run_dir)


def remove_tree_with_retry(path: Path, retries: int = 3, wait_seconds: float = 0.2) -> bool:
    if not path.exists():
        return True

    def _onerror(func, p, exc_info):
        try:
            os.chmod(p, 0o777)
            func(p)
        except Exception:
            pass

    for _ in range(retries):
        try:
            shutil.rmtree(path, onerror=_onerror)
            if not path.exists():
                return True
        except Exception:
            pass
        time.sleep(wait_seconds)
    return not path.exists()


def clear_all_history_files(history: List[dict]) -> tuple[int, List[str]]:
    deleted_count = 0
    failed_paths: List[str] = []
    for item in history:
        run_dir_text = str(item.get("run_dir", "")).strip()
        if not run_dir_text:
            continue
        run_dir = Path(run_dir_text)
        ok = remove_tree_with_retry(run_dir)
        if ok:
            deleted_count += 1
        else:
            failed_paths.append(str(run_dir))

    if WEB_RUNS_DIR.exists():
        for child in WEB_RUNS_DIR.iterdir():
            if child.is_dir():
                ok = remove_tree_with_retry(child)
                if not ok:
                    failed_paths.append(str(child))

    return deleted_count, failed_paths


def persist_run_outputs(
    produced_files: List[Path],
    stdout_text: str,
    stderr_text: str,
) -> dict:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = WEB_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    copied_files: List[str] = []
    for file_path in produced_files:
        target = run_dir / file_path.name
        shutil.copy2(file_path, target)
        copied_files.append(target.name)

    (run_dir / "stdout.log").write_text(stdout_text or "", encoding="utf-8")
    (run_dir / "stderr.log").write_text(stderr_text or "", encoding="utf-8")

    return {
        "run_id": run_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": str(run_dir),
        "output_files": copied_files,
    }


def main() -> None:
    st.set_page_config(page_title="智能特征分箱分析平台", layout="wide")
    inject_ui_style()
    st.title("智能特征分箱分析平台")
    st.caption("上传数据文件，选择参数和配置后生成分析结果。")
    run_tab, history_tab = st.tabs(["运行分析", "任务历史"])

    with history_tab:
        history = load_web_history()
        if not history:
            st.info("暂无历史任务。")
        else:
            st.caption(f"共 {len(history)} 条记录（最新在前）")
            if st.button("清空全部历史", key="clear_all_history", use_container_width=True):
                deleted_count, failed_paths = clear_all_history_files(history)
                save_web_history([])
                if failed_paths:
                    st.warning(f"历史索引已清空，但有 {len(failed_paths)} 个目录删除失败（可能被占用）。")
                    st.code("\n".join(failed_paths[:20]), language="text")
                else:
                    st.success(f"已清空全部历史记录（删除目录 {deleted_count} 个）。")
                st.rerun()
            for idx, item in enumerate(history, start=1):
                run_dir = Path(item.get("run_dir", ""))
                title = f"{idx}. {item.get('created_at', '-')} | {item.get('input_file', '-')}"
                with st.expander(title, expanded=False):
                    st.write(f"status_source: {item.get('status_source', '-')}")
                    st.write(f"输出文件数: {item.get('output_count', 0)}")
                    output_files = item.get("output_files", [])
                    if not output_files and run_dir.exists():
                        output_files = [p.name for p in sorted(run_dir.glob("feature_analysis*.xlsx"))]
                    if output_files:
                        for file_name in output_files:
                            excel_path = run_dir / file_name
                            if excel_path.exists():
                                st.download_button(
                                    f"下载 Excel（{file_name}）",
                                    data=excel_path.read_bytes(),
                                    file_name=file_name,
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key=f"history_excel_{idx}_{file_name}",
                                )
                    else:
                        st.warning("历史 Excel 文件不存在，可能已被手动清理。")
                    stdout_log = run_dir / "stdout.log"
                    stderr_log = run_dir / "stderr.log"
                    if stdout_log.exists():
                        st.code(stdout_log.read_text(encoding="utf-8"), language="text")
                    if stderr_log.exists() and stderr_log.read_text(encoding="utf-8").strip():
                        st.code(stderr_log.read_text(encoding="utf-8"), language="text")
                    if st.button("删除该条历史", key=f"delete_history_{idx}"):
                        new_history = [h for h in history if h.get("run_id") != item.get("run_id")]
                        delete_history_item_files(item)
                        save_web_history(new_history)
                        st.success("该条历史已删除。")
                        st.rerun()

    with run_tab:
        with st.container(border=True):
            uploaded_data = st.file_uploader(
                "上传数据文件",
                type=["xlsx", "xls", "xlsm", "csv", "parquet"],
            )
            status_source = st.selectbox(
                "起始 settleStatus 列",
                options=[0, 1, 2, 3],
                format_func=lambda x: {
                    0: "0 -> settleStatus",
                    1: "1 -> settleStatus_1",
                    2: "2 -> settleStatus_2",
                    3: "3 -> settleStatus_3",
                }[x],
                index=0,
            )
            with st.expander("高级筛选（可选）", expanded=False):
                feature_prefixes = st.text_input("特征前缀过滤（逗号分隔）", value="")

        st.markdown("#### 配置选择")
        config_col1, config_col2, config_col3 = st.columns([1.2, 1, 1.2])

        desc_candidates = list_config_files(["*.xlsx", "*.xls"])
        sheet_toggle_defaults = load_sheet_toggle_options()
        sheet_toggle_values: dict = {}

        with config_col1:
            with st.container(border=True):
                st.caption("Sheet 输出开关（逐项）")
                if sheet_toggle_defaults:
                    with st.expander("展开 Sheet 开关", expanded=False):
                        for sheet_name in sorted(sheet_toggle_defaults.keys(), key=str.upper):
                            sheet_toggle_values[sheet_name] = st.toggle(
                                sheet_name,
                                value=bool(sheet_toggle_defaults[sheet_name]),
                                key=f"sheet_toggle_{sheet_name}",
                            )
                else:
                    st.info("未找到 Sheet 配置文件，将按脚本默认规则输出。")
        with config_col2:
            with st.container(border=True):
                st.caption("展示配置")
                show_feature_desc = st.toggle("显示特征中文解释", value=True)
        with config_col3:
            with st.container(border=True):
                desc_options = ["默认"] + [str(p) for p in desc_candidates]
                desc_choice = st.selectbox("特征中文解释表", desc_options, index=0)
                desc_upload = st.file_uploader("或上传特征解释 Excel", type=["xlsx", "xls"], key="desc_xlsx")
                st.download_button(
                    "下载示例文件",
                    data=build_feature_desc_template_bytes(),
                    file_name="特征中文解释表示例_模板.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_desc_sample",
                    use_container_width=True,
                )

        run_btn = st.button("开始分析", type="primary", use_container_width=True)
        if not run_btn:
            return
        if not uploaded_data:
            st.error("请先上传数据文件。")
            return

    run_dir: Optional[Path] = None
    try:
        run_dir = create_writable_run_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_name = safe_upload_filename(uploaded_data.name, fallback="input_file")
        input_path = save_uploaded_file(
            uploaded_data,
            run_dir / input_name,
            fallback_dirs=[run_dir.parent],
        )
        output_base_path = run_dir / f"feature_analysis_{timestamp}.xlsx"

        sheet_config_path: Optional[Path] = None
        display_config_path: Optional[Path] = None
        feature_desc_path: Optional[Path] = None

        if sheet_toggle_values:
            sheet_config_path = write_json_config(
                sheet_toggle_values,
                run_dir / "feature_sheet_output_config.json",
            )
        else:
            default_sheet_config = CONFIG_DIR / "feature_sheet_output_config.json"
            if default_sheet_config.exists():
                sheet_config_path = default_sheet_config

        display_config_path = write_json_config(
            {"show_feature_desc": bool(show_feature_desc)},
            run_dir / "feature_display_config.json",
        )

        if desc_upload:
            desc_name = safe_upload_filename(desc_upload.name, fallback="feature_desc.xlsx")
            feature_desc_path = save_uploaded_file(
                desc_upload,
                run_dir / desc_name,
                fallback_dirs=[run_dir.parent],
            )
        elif desc_choice != "默认":
            feature_desc_path = Path(desc_choice)

        st.subheader("运行进度")
        progress_placeholder = st.empty()
        stage_placeholder = st.empty()
        log_placeholder = st.empty()
        return_code, stdout_text, stderr_text = run_analysis_with_live_progress(
            input_path=input_path,
            output_base_path=output_base_path,
            status_source=status_source,
            feature_prefixes=feature_prefixes,
            sheet_config_path=sheet_config_path,
            display_config_path=display_config_path,
            feature_desc_path=feature_desc_path,
            progress_placeholder=progress_placeholder,
            stage_placeholder=stage_placeholder,
            log_placeholder=log_placeholder,
        )

        if stderr_text.strip():
            st.error("运行过程中出现错误输出，请查看下方详情。")
            st.code(stderr_text, language="text")

        if return_code != 0:
            st.error(f"运行失败，返回码：{return_code}")
            return

        produced_files = sorted(run_dir.glob("feature_analysis*.xlsx"))
        if not produced_files:
            st.error("运行成功，但未找到输出 Excel。")
            return

        persisted = persist_run_outputs(
            produced_files=produced_files,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
        )
        history = load_web_history()
        history.insert(
            0,
            {
                "created_at": persisted["created_at"],
                "run_id": persisted["run_id"],
                "input_file": input_name,
                "status_source": status_source,
                "output_count": len(produced_files),
                "output_files": persisted["output_files"],
                "run_dir": persisted["run_dir"],
            },
        )
        save_web_history(history[:100])

        st.success(f"分析完成，共生成 {len(produced_files)} 个 Excel 文件，已记录到历史。")
        st.info(f"结果目录：{persisted['run_dir']}")
        st.subheader("下载结果（Excel）")
        for idx, file_path in enumerate(produced_files, start=1):
            st.download_button(
                f"下载 Excel {idx}: {file_path.name}",
                data=file_path.read_bytes(),
                file_name=file_path.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_excel_{idx}_{file_path.name}",
                use_container_width=True,
            )
    finally:
        if run_dir is not None:
            pass


if __name__ == "__main__":
    main()
