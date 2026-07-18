from __future__ import annotations

import numpy as np

from app.services.face_analysis import (
    analyze_faces,
)


def cosine_similarity(
    first_embedding: np.ndarray,
    second_embedding: np.ndarray,
) -> float:
    first = np.asarray(
        first_embedding,
        dtype=np.float32,
    )

    second = np.asarray(
        second_embedding,
        dtype=np.float32,
    )

    if first.shape != second.shape:
        raise ValueError(
            "Face embeddings have "
            "different shapes."
        )

    first_norm = float(
        np.linalg.norm(first)
    )

    second_norm = float(
        np.linalg.norm(second)
    )

    if first_norm <= 0:
        raise ValueError(
            "Source embedding "
            "has zero norm."
        )

    if second_norm <= 0:
        raise ValueError(
            "Result embedding "
            "has zero norm."
        )

    score = float(
        np.dot(
            first / first_norm,
            second / second_norm,
        )
    )

    return round(
        float(
            np.clip(
                score,
                -1.0,
                1.0,
            )
        ),
        4,
    )


def compare_source_with_result(
    *,
    source_embedding: np.ndarray,
    result_image_path: str,
) -> float:
    result_analysis = analyze_faces(
        result_image_path,
        allow_enhancement=True,
    )

    return cosine_similarity(
        source_embedding,
        result_analysis
        .primary_face
        .normalized_embedding,
    )