from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

from app.core.config import settings


def _tail_text(path: Path, max_bytes: int = 24000) -> str:
    if not path.is_file():
        return ""

    with path.open("rb") as source:
        source.seek(0, os.SEEK_END)
        size = source.tell()
        source.seek(max(0, size - max_bytes))
        data = source.read()

    return data.decode("utf-8", errors="replace").strip()


def _terminate_process_group(process: subprocess.Popen[object]) -> None:
    if process.poll() is not None:
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def apply_birefnet_portrait_background(
    *,
    input_path: str,
    output_path: str,
    job_directory: Path,
) -> None:
    if not settings.background_matting_enabled:
        raise RuntimeError(
            "Background matting is disabled."
        )

    script_path = Path(
        settings.background_matting_script_path
    )

    if not script_path.is_file():
        raise FileNotFoundError(
            "Background matting script does not exist: "
            f"{script_path}"
        )

    source_path = Path(input_path).resolve()
    final_path = Path(output_path).resolve()

    if not source_path.is_file():
        raise FileNotFoundError(
            "Selected portrait does not exist: "
            f"{source_path}"
        )

    job_directory = job_directory.resolve()

    if source_path.parent != job_directory:
        raise ValueError(
            "Selected portrait is outside the job directory."
        )

    if final_path.parent != job_directory:
        raise ValueError(
            "Final portrait path is outside the job directory."
        )

    temporary_output_path = (
        job_directory / "result_background.tmp.png"
    )
    mask_path = job_directory / "background_mask.png"
    stats_path = job_directory / "background_matting_stats.json"
    log_path = job_directory / "birefnet_portrait.log"

    command = [
        sys.executable,
        str(script_path),
        "--input",
        str(source_path),
        "--output",
        str(temporary_output_path),
        "--model",
        settings.background_matting_model,
        "--background-hex",
        settings.corporate_background_hex,
        "--alpha-gamma",
        str(settings.background_matting_alpha_gamma),
        "--min-foreground-ratio",
        str(settings.background_matting_min_foreground_ratio),
        "--max-foreground-ratio",
        str(settings.background_matting_max_foreground_ratio),
        "--stats-output",
        str(stats_path),
    ]

    if settings.keep_background_mask:
        command.extend(
            [
                "--mask-output",
                str(mask_path),
            ]
        )

    environment = os.environ.copy()
    environment.update(
        {
            "U2NET_HOME": settings.background_matting_model_dir,
            "OMP_NUM_THREADS": str(
                settings.background_matting_threads
            ),
        }
    )

    process: subprocess.Popen[object] | None = None

    try:
        temporary_output_path.unlink(missing_ok=True)

        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd="/app",
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

            try:
                return_code = process.wait(
                    timeout=(
                        settings
                        .background_matting_timeout_seconds
                    )
                )
            except subprocess.TimeoutExpired as exc:
                _terminate_process_group(process)
                raise TimeoutError(
                    "BiRefNet-portrait processing exceeded "
                    f"{settings.background_matting_timeout_seconds} "
                    f"seconds. Log: {log_path}"
                ) from exc

        if return_code != 0:
            log_tail = _tail_text(log_path)
            raise RuntimeError(
                "BiRefNet-portrait processing failed with "
                f"exit code {return_code}. "
                f"Log tail: {log_tail or '<empty>'}"
            )

        if (
            not temporary_output_path.is_file()
            or temporary_output_path.stat().st_size <= 0
        ):
            raise RuntimeError(
                "BiRefNet-portrait did not create a valid "
                f"result: {temporary_output_path}"
            )

        temporary_output_path.replace(final_path)

    finally:
        if process is not None and process.poll() is None:
            _terminate_process_group(process)

        temporary_output_path.unlink(missing_ok=True)

        if not settings.keep_background_mask:
            mask_path.unlink(missing_ok=True)