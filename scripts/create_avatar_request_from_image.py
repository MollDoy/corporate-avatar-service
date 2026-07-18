import base64
import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage: "
            "python "
            "scripts/create_avatar_request_from_image.py "
            "<image_path> <employee_id> "
            "[style_id]"
        )

    image_path = Path(
        sys.argv[1]
    )

    employee_id = sys.argv[2]

    style_id = (
        sys.argv[3]
        if len(sys.argv) >= 4
        else "ai_business"
    )

    if not image_path.is_file():
        raise SystemExit(
            f"Image does not exist: "
            f"{image_path}"
        )

    image_base64 = base64.b64encode(
        image_path.read_bytes()
    ).decode("ascii")

    payload = {
        "employee_id": employee_id,
        "style_id": style_id,
        "image_base64": image_base64,
    }

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()