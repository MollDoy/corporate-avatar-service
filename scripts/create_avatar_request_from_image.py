import base64
import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python3 scripts/create_avatar_request_from_image.py "
            "<image_path> [employee_id] [style_id]",
            file=sys.stderr,
        )
        raise SystemExit(1)

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"File not found: {image_path}", file=sys.stderr)
        raise SystemExit(1)

    employee_id = sys.argv[2] if len(sys.argv) >= 3 else "employee_001"
    style_id = sys.argv[3] if len(sys.argv) >= 4 else "default_business"

    image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    payload = {
        "employee_id": employee_id,
        "style_id": style_id,
        "image_base64": image_base64,
    }

    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()