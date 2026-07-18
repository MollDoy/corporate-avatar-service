from __future__ import annotations

import argparse
import py_compile
from pathlib import Path


class PatchError(RuntimeError):
    pass


def _method_bounds(
    lines: list[str],
    method_name: str,
) -> tuple[int, int, str]:
    definition_index: int | None = None

    for index, line in enumerate(lines):
        if line.lstrip().startswith(f"def {method_name}("):
            definition_index = index
            break

    if definition_index is None:
        raise PatchError(f"Method was not found: {method_name}")

    indent = lines[definition_index][
        : len(lines[definition_index])
        - len(lines[definition_index].lstrip())
    ]

    start_index = definition_index
    previous_index = definition_index - 1

    while previous_index >= 0:
        previous = lines[previous_index]

        if not previous.strip():
            previous_index -= 1
            continue

        if (
            previous.startswith(indent)
            and previous.strip().startswith("@")
        ):
            start_index = previous_index

        break

    end_index = len(lines)

    for index in range(definition_index + 1, len(lines)):
        line = lines[index]

        if not line.strip():
            continue

        current_indent = line[
            : len(line) - len(line.lstrip())
        ]

        if (
            current_indent == indent
            and (
                line.lstrip().startswith("def ")
                or line.lstrip().startswith("@")
            )
        ):
            end_index = index
            break

    return start_index, end_index, indent


def _replace_method(
    lines: list[str],
    method_name: str,
    replacement: str,
) -> list[str]:
    start, end, indent = _method_bounds(
        lines,
        method_name,
    )

    replacement_lines = [
        indent + line if line else ""
        for line in replacement.strip("\n").splitlines()
    ]

    return (
        lines[:start]
        + replacement_lines
        + [""]
        + lines[end:]
    )


def _replace_once(
    text: str,
    old: str,
    new: str,
    *,
    label: str,
) -> str:
    count = text.count(old)

    if count != 1:
        raise PatchError(
            f"{label}: expected one occurrence, found {count}."
        )

    return text.replace(old, new, 1)


def patch_pipeline(source_path: Path) -> None:
    if not source_path.is_file():
        raise FileNotFoundError(
            "ConsistentID pipeline source does not exist: "
            f"{source_path}"
        )

    lines = source_path.read_text(
        encoding="utf-8"
    ).splitlines()

    insightface_imports = [
        index
        for index, line in enumerate(lines)
        if line.strip()
        == "from insightface.app import FaceAnalysis"
    ]

    if len(insightface_imports) != 1:
        raise PatchError(
            "Expected exactly one internal FaceAnalysis import, "
            f"found {len(insightface_imports)}."
        )

    del lines[insightface_imports[0]]

    app_indexes = [
        index
        for index, line in enumerate(lines)
        if "self.app = FaceAnalysis(" in line
    ]

    prepare_indexes = [
        index
        for index, line in enumerate(lines)
        if "self.app.prepare(" in line
    ]

    if len(app_indexes) != 1 or len(prepare_indexes) != 1:
        raise PatchError(
            "Could not locate the internal ConsistentID "
            "FaceAnalysis block."
        )

    app_index = app_indexes[0]
    prepare_index = prepare_indexes[0]

    if prepare_index != app_index + 1:
        raise PatchError(
            "Unexpected FaceAnalysis initialization layout."
        )

    indent = lines[app_index][
        : len(lines[app_index])
        - len(lines[app_index].lstrip())
    ]

    lines[app_index : prepare_index + 1] = [
        indent + "# FaceID is supplied by the worker's buffalo_l model.",
        indent + "self.external_faceid_embeds = None",
    ]

    lines = _replace_method(
        lines,
        "get_prepare_faceid",
        '''@torch.inference_mode()
def get_prepare_faceid(self, face_image):
    del face_image

    if self.external_faceid_embeds is None:
        raise RuntimeError(
            "ConsistentID external_faceid_embeds was not provided."
        )

    faceid_embeds = (
        self.external_faceid_embeds
        .detach()
        .clone()
        .to(device="cpu", dtype=torch.float32)
    )

    if faceid_embeds.ndim == 1:
        faceid_embeds = faceid_embeds.unsqueeze(0)

    if faceid_embeds.shape != (1, 512):
        raise ValueError(
            "ConsistentID external face embedding must have "
            "shape (1, 512), got "
            f"{tuple(faceid_embeds.shape)}."
        )

    return faceid_embeds''',
    )

    text = "\n".join(lines) + "\n"

    text = _replace_once(
        text,
        "self.bise_net.cuda()",
        'self.bise_net.to(device="cpu", dtype=torch.float32)',
        label="initialize BiSeNet on CPU",
    )

    text = _replace_once(
        text,
        "self.bise_net.load_state_dict(torch.load(self.bise_net_cp))",
        (
            "self.bise_net.load_state_dict(\n"
            "            torch.load(\n"
            "                self.bise_net_cp,\n"
            "                map_location=\"cpu\",\n"
            "            )\n"
            "        )"
        ),
        label="load BiSeNet checkpoint on CPU",
    )

    text = _replace_once(
        text,
        "img = img.float().cuda()",
        (
            "img = img.float().to(\n"
            "                next(self.bise_net.parameters()).device\n"
            "            )"
        ),
        label="use the active BiSeNet device",
    )

    text = _replace_once(
        text,
        "for attn_processor in self.pipe.unet.attn_processors.values():",
        "for attn_processor in self.unet.attn_processors.values():",
        label="fix official set_scale UNet reference",
    )

    source_path.write_text(
        text,
        encoding="utf-8",
    )

    py_compile.compile(
        str(source_path),
        doraise=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply only the minimal deployment compatibility patch "
            "to the official ConsistentID SD 1.5 pipeline."
        )
    )

    parser.add_argument(
        "pipeline_path",
        type=Path,
    )

    arguments = parser.parse_args()

    patch_pipeline(
        arguments.pipeline_path.resolve()
    )

    print(
        "[consistentid-patch] Minimal official-pipeline patch "
        "applied successfully.",
        flush=True,
    )


if __name__ == "__main__":
    main()