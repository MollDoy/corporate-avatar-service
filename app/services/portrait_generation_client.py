from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings


@dataclass(frozen=True)
class GeneratedCandidate:
    image_path: str
    seed: int
    attempt_number: int


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


def generate_portrait_candidates(
    *,
    job_id: str,
    identity_reference_name: str,
    face_embedding_name: str,
    prompt: str,
    negative_prompt: str,
    seeds: list[int],
    attempt_number_start: int = 1,
) -> list[GeneratedCandidate]:
    script_path = Path(settings.ai_batch_script_path)

    if not script_path.is_file():
        raise FileNotFoundError(
            "ConsistentID batch script does not exist: "
            f"{script_path}"
        )

    job_directory = (
        Path(settings.storage_dir)
        / "jobs"
        / job_id
    ).resolve()

    if not job_directory.is_dir():
        raise FileNotFoundError(
            "Job directory does not exist: "
            f"{job_directory}"
        )

    if attempt_number_start < 1:
        raise ValueError(
            "attempt_number_start must be greater than zero."
        )

    requests: list[dict[str, object]] = []
    expected_candidates: list[GeneratedCandidate] = []

    for offset, seed in enumerate(seeds):
        attempt_number = attempt_number_start + offset
        output_name = (
            f"candidate_{attempt_number:02d}_seed_{seed}.png"
        )

        requests.append(
            {
                "job_id": job_id,
                "identity_reference_name": identity_reference_name,
                "face_embedding_name": face_embedding_name,
                "output_name": output_name,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": seed,
                "num_inference_steps": (
                    settings.ai_num_inference_steps
                ),
                "guidance_scale": settings.ai_guidance_scale,
                "adapter_scale": (
                    settings.ai_consistentid_adapter_scale
                ),
                "start_merge_step": (
                    settings.ai_consistentid_start_merge_step
                ),
            }
        )

        expected_candidates.append(
            GeneratedCandidate(
                image_path=str(
                    job_directory / output_name
                ),
                seed=seed,
                attempt_number=attempt_number,
            )
        )

    manifest_path = (
        job_directory / "consistentid_batch_request.json"
    )
    response_path = (
        job_directory / "consistentid_batch_response.json"
    )
    log_path = job_directory / "consistentid_batch.log"

    manifest = {
        "requests": requests,
        "response_path": str(response_path),
    }

    temporary_manifest = manifest_path.with_suffix(
        ".json.tmp"
    )

    temporary_manifest.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary_manifest.replace(manifest_path)

    environment = os.environ.copy()

    environment.update(
        {
            "STORAGE_ROOT": str(
                Path(settings.storage_dir) / "jobs"
            ),
            "AI_OUTPUT_SIZE": str(
                settings.ai_output_size
            ),
            "AI_CPU_THREADS": str(
                settings.ai_cpu_threads
            ),
        }
    )

    command = [
        sys.executable,
        str(script_path),
        "--batch-manifest",
        str(manifest_path),
    ]

    process: subprocess.Popen[object] | None = None

    try:
        with log_path.open(
            "w",
            encoding="utf-8",
        ) as log_file:
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
                    timeout=settings.ai_timeout_seconds
                )
            except subprocess.TimeoutExpired as exc:
                _terminate_process_group(process)

                raise TimeoutError(
                    "ConsistentID batch generation exceeded "
                    f"{settings.ai_timeout_seconds} seconds. "
                    f"Log: {log_path}"
                ) from exc

        if return_code != 0:
            log_tail = _tail_text(log_path)

            raise RuntimeError(
                "ConsistentID batch generation failed with "
                f"exit code {return_code}. "
                f"Log tail: {log_tail or '<empty>'}"
            )

        if not response_path.is_file():
            raise RuntimeError(
                "ConsistentID batch process did not create its "
                f"response file: {response_path}"
            )

        try:
            response_data = json.loads(
                response_path.read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "ConsistentID batch response is invalid."
            ) from exc

        returned_results = response_data.get(
            "results"
        )

        if not isinstance(
            returned_results,
            list,
        ):
            raise RuntimeError(
                "ConsistentID batch response does not contain "
                "a results list."
            )

        returned_by_seed: dict[int, str] = {}

        for item in returned_results:
            if not isinstance(item, dict):
                continue

            seed = item.get("seed")
            output_path = item.get(
                "output_path"
            )

            if (
                isinstance(seed, int)
                and isinstance(output_path, str)
            ):
                returned_by_seed[seed] = (
                    output_path
                )

        for candidate in expected_candidates:
            expected_path = Path(
                candidate.image_path
            ).resolve()

            returned_path_raw = (
                returned_by_seed.get(
                    candidate.seed
                )
            )

            if returned_path_raw is None:
                raise RuntimeError(
                    "ConsistentID batch response omitted seed "
                    f"{candidate.seed}."
                )

            returned_path = Path(
                returned_path_raw
            ).resolve()

            if returned_path != expected_path:
                raise RuntimeError(
                    "ConsistentID batch returned an unexpected "
                    f"path for seed {candidate.seed}: "
                    f"{returned_path}"
                )

            if (
                not expected_path.is_file()
                or expected_path.stat().st_size <= 0
            ):
                raise RuntimeError(
                    "ConsistentID did not create a valid candidate: "
                    f"{expected_path}"
                )

        return expected_candidates

    finally:
        if (
            process is not None
            and process.poll() is None
        ):
            _terminate_process_group(
                process
            )

        manifest_path.unlink(
            missing_ok=True
        )

        response_path.unlink(
            missing_ok=True
        )