"""CPU inference for an outage of all free hosted routes, using pinned public assets."""
import hashlib
import logging
import os
import subprocess
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

logger = logging.getLogger(__name__)
MODEL_NAME = "Qwen2.5-1.5B-Instruct-Q4_K_M"
BUILD = "b12676"
RUNTIME_URL = (
    f"https://github.com/ggml-org/llama.cpp/releases/download/{BUILD}/"
    f"llama-{BUILD}-bin-ubuntu-x64.tar.gz"
)
MODEL_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/"
    "91cad51170dc346986eccefdc2dd33a9da36ead9/qwen2.5-1.5b-instruct-q4_k_m.gguf"
)


def download(url: str, path: Path, digest: str) -> None:
    subprocess.run(
        ["curl", "--disable", "--fail", "--silent", "--show-error", "--location",
         "--proto", "=https", "--proto-redir", "=https", "--connect-timeout", "15",
         "--max-time", "240", "--retry", "2", "--output", str(path), url],
        check=True, timeout=480,
    )
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
            raise ValueError(f"SHA-256 mismatch for {path.name}")


class CpuForecaster:
    def __init__(self):
        self.temp = TemporaryDirectory(prefix="futureeval-model-")
        self.root = Path(self.temp.name)
        archive = self.root / "runtime.tar.gz"
        download(RUNTIME_URL, archive, "c65369a8574d096d90c593fef0e77a1ba2abfdfa7e7e9426bbf8939781ef8a07")
        with tarfile.open(archive) as runtime:
            runtime.extractall(self.root, filter="data")
        self.executable = self.root / f"llama-{BUILD}" / "llama-completion"
        self.weights = self.root / "model.gguf"
        download(MODEL_URL, self.weights, "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e")
        logger.info("Verified CPU model weights and inference runtime")

    def invoke(self, prompt: str, system_prompt: str | None = None) -> str:
        system = system_prompt or "Follow the instructions and use the requested final answer format."
        prompt_file = self.root / "prompt.txt"
        prompt_file.write_text(
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [self.executable, "-m", self.weights, "-f", prompt_file, "--no-conversation",
             "--no-display-prompt", "--no-warmup", "--log-disable", "--simple-io",
             "-ngl", "0", "-t", str(min(4, os.cpu_count() or 1)), "-c", "8192",
             "-n", "768", "--temp", "0.2", "-r", "<|im_end|>"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True, check=True,
            timeout=240, env={"PATH": os.defpath, "HOME": str(self.root)},
        )
        answer = result.stdout.strip()
        if not answer:
            raise ValueError("The CPU model returned an empty response")
        logger.info("Model responded: %s (local CPU)", MODEL_NAME)
        return answer
