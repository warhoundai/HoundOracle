"""Short-lived local inference worker; input/output are bounded JSON over stdio."""

import base64
import io
import json
import math
import socket
import subprocess
import sys
import warnings


def run(request):
    def network_disabled(*args, **kwargs):
        raise OSError("CAPTCHA_WORKER_NETWORK_DISABLED")

    # Defense in depth for Python dependencies; process boundary is not a full sandbox.
    socket.socket.connect = network_disabled
    socket.socket.connect_ex = network_disabled
    socket.create_connection = network_disabled
    socket.getaddrinfo = network_disabled
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (math.ceil(request["seconds"]) + 1,) * 2)
        resource.setrlimit(resource.RLIMIT_FSIZE, (2_000_000,) * 2)
    except ImportError:
        pass
    try:
        from PIL import Image
    except ImportError:
        return {"status": "BACKEND_UNAVAILABLE"}
    Image.MAX_IMAGE_PIXELS = 1_000_000
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            raw = base64.b64decode(request["image"], validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format not in {"PNG", "JPEG"} or getattr(image, "n_frames", 1) != 1:
                    return {"status": "INVALID_IMAGE"}
                if (
                    not 8 <= image.width <= 4096
                    or not 8 <= image.height <= 4096
                    or image.width * image.height > 1_000_000
                    or image.width / image.height > 16
                    or image.height / image.width > 16
                ):
                    return {"status": "INVALID_IMAGE"}
                image = image.convert("RGB")
                stream = io.BytesIO()
                image.save(stream, format="PNG")
                raw = stream.getvalue()
    except Exception:  # noqa: BLE001 -- isolate malformed images and native model failures
        return {"status": "INVALID_IMAGE"}
    if request["backend"] == "ddddocr":
        try:
            import contextlib
            from importlib.metadata import version

            # Upstream banners/logs must not corrupt the machine-readable response.
            with contextlib.redirect_stdout(io.StringIO()):
                import ddddocr

                model = ddddocr.DdddOcr(ocr=True, det=False, use_gpu=False, show_ad=False)
                import string

                model.set_ranges(string.ascii_letters + string.digits)
                candidate = model.classification(raw)
                # Preserve OCR characters before projecting to an explicitly declared
                # one-case field alphabet. Never change case for mixed-case fields.
                candidate = "".join(
                    c
                    if c in request["alphabet"]
                    else (c.swapcase() if c.swapcase() in request["alphabet"] else c)
                    for c in candidate
                )
            backend_version = version("ddddocr")
        except ImportError:
            return {"status": "BACKEND_UNAVAILABLE"}
        except Exception:  # noqa: BLE001 -- isolate malformed images and native model failures
            return {"status": "BACKEND_FAILED"}
    else:
        try:
            output = subprocess.run(
                [
                    "tesseract",
                    "stdin",
                    "stdout",
                    "--psm",
                    "7",
                    "-c",
                    "tessedit_char_whitelist=" + request["alphabet"],
                ],
                input=raw,
                capture_output=True,
                check=True,
                timeout=request["seconds"],
            )
            candidate = "".join(output.stdout.decode("utf-8", "replace").split())
            backend_version = "system-tesseract"
        except FileNotFoundError:
            return {"status": "BACKEND_UNAVAILABLE"}
        except (subprocess.SubprocessError, OSError):
            return {"status": "BACKEND_FAILED"}
    return {
        "status": "CANDIDATE" if candidate else "NO_CANDIDATE",
        "candidate": candidate or None,
        "backend_version": backend_version,
    }


if __name__ == "__main__":
    try:
        request = json.loads(sys.stdin.buffer.read(1_400_000))
        result = run(request)
    except Exception:  # noqa: BLE001 -- isolate malformed images and native model failures
        result = {"status": "BACKEND_FAILED"}
    sys.stdout.write(json.dumps(result))
