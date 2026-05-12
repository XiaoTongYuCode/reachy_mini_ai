"""Manage an optional local Hugging Face realtime gateway subprocess."""

from __future__ import annotations
import os
import signal
import asyncio
import logging
import subprocess
from pathlib import Path
from collections.abc import Mapping, Sequence

from reachy_mini_conversation_app.config import PROJECT_ROOT, HF_REALTIME_AUTO_START_TIMEOUT_SECONDS_ENV


logger = logging.getLogger(__name__)

DEFAULT_GATEWAY_COMMAND = "reachy-mini-hf-realtime-gateway"
DEFAULT_GATEWAY_READY_TIMEOUT_SECONDS = 600.0


class HFRealtimeGatewayProcess:
    """Start and stop a locally managed `reachy-mini-hf-realtime-gateway` process."""

    def __init__(
        self,
        command: Sequence[str] | None = None,
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        ready_timeout_seconds: float = DEFAULT_GATEWAY_READY_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize the managed gateway command and readiness timeout."""
        if command is None:
            self.command, self.cwd, self.env = _default_gateway_launch()
        else:
            self.command = list(command)
            self.cwd = Path(cwd) if cwd is not None else None
            self.env = dict(env) if env is not None else None
        self.ready_timeout_seconds = _resolve_ready_timeout(ready_timeout_seconds)
        self._process: subprocess.Popen[bytes] | None = None

    async def ensure_started(self, realtime_url: str) -> None:
        """Start the gateway if needed and wait until its websocket accepts connections."""
        if self.is_running:
            return

        try:
            self._process = subprocess.Popen(
                self.command,
                cwd=self.cwd,
                env=self.env,
                start_new_session=os.name != "nt",
            )
        except FileNotFoundError as exc:
            command_name = self.command[0] if self.command else DEFAULT_GATEWAY_COMMAND
            raise RuntimeError(
                f"{command_name!r} was not found. Install services/hf_realtime_gateway or put it on PATH "
                "before setting HF_REALTIME_AUTO_START=true."
            ) from exc
        except OSError as exc:
            raise RuntimeError(f"Failed to start local Hugging Face realtime gateway: {exc}") from exc

        logger.info("Started local Hugging Face realtime gateway process pid=%s", self._process.pid)
        logger.info(
            "Waiting up to %.0f seconds for local Hugging Face realtime gateway readiness at %s",
            self.ready_timeout_seconds,
            realtime_url,
        )
        try:
            await self._wait_until_ready(realtime_url)
        except Exception:
            await self.stop()
            raise

    @property
    def is_running(self) -> bool:
        """Return whether the managed gateway process is still running."""
        return self._process is not None and self._process.poll() is None

    async def _wait_until_ready(self, realtime_url: str) -> None:
        """Wait for the realtime websocket to become reachable."""
        deadline = asyncio.get_running_loop().time() + self.ready_timeout_seconds
        last_error: BaseException | None = None
        while asyncio.get_running_loop().time() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    "Local Hugging Face realtime gateway exited before it became ready "
                    f"(exit code {self._process.returncode})."
                )
            try:
                await _probe_realtime_websocket(realtime_url)
                logger.info("Local Hugging Face realtime gateway is ready at %s", realtime_url)
                return
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(0.5)

        detail = f": {last_error}" if last_error is not None else ""
        raise RuntimeError(
            f"Timed out waiting for local Hugging Face realtime gateway at {realtime_url}{detail}"
        )

    async def stop(self) -> None:
        """Terminate the managed gateway process and its child process group."""
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return

        logger.info("Stopping local Hugging Face realtime gateway process pid=%s", process.pid)
        try:
            if os.name != "nt":
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
                process.terminate()
            await asyncio.to_thread(process.wait, 10)
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:
                process.kill()
            await asyncio.to_thread(process.wait)
        except ProcessLookupError:
            return


async def _probe_realtime_websocket(realtime_url: str) -> None:
    """Open and close the realtime websocket as a lightweight readiness probe."""
    import websockets

    async with websockets.connect(
        realtime_url,
        open_timeout=1.0,
        close_timeout=1.0,
    ):
        return


def _default_gateway_launch() -> tuple[list[str], Path | None, dict[str, str] | None]:
    """Return the local checkout gateway launch config when present, otherwise PATH command."""
    service_root = PROJECT_ROOT / "services" / "hf_realtime_gateway"
    service_venv_bin = service_root / ".venv" / "bin"
    local_command = service_venv_bin / DEFAULT_GATEWAY_COMMAND
    if local_command.is_file():
        command = [str(local_command)]
        env_file = service_root / ".env"
        if env_file.is_file():
            command.extend(["--env-file", str(env_file)])
        env = dict(os.environ)
        if not _env_file_defines(env_file, "HF_HOME"):
            env.pop("HF_HOME", None)
        env["PATH"] = os.pathsep.join([str(service_venv_bin), env.get("PATH", "")])
        return command, service_root, env
    return [DEFAULT_GATEWAY_COMMAND], None, None


def _resolve_ready_timeout(default_timeout_seconds: float) -> float:
    """Resolve the managed gateway readiness timeout from the environment."""
    raw_value = os.environ.get(HF_REALTIME_AUTO_START_TIMEOUT_SECONDS_ENV)
    if raw_value is None or not raw_value.strip():
        return default_timeout_seconds
    try:
        timeout_seconds = float(raw_value)
    except ValueError:
        logger.warning(
            "Ignoring invalid %s=%r; using %.0f seconds.",
            HF_REALTIME_AUTO_START_TIMEOUT_SECONDS_ENV,
            raw_value,
            default_timeout_seconds,
        )
        return default_timeout_seconds
    if timeout_seconds <= 0:
        logger.warning(
            "Ignoring non-positive %s=%r; using %.0f seconds.",
            HF_REALTIME_AUTO_START_TIMEOUT_SECONDS_ENV,
            raw_value,
            default_timeout_seconds,
        )
        return default_timeout_seconds
    return timeout_seconds


def _env_file_defines(env_file: Path, key: str) -> bool:
    """Return whether a dotenv file explicitly defines a key."""
    if not env_file.is_file():
        return False
    try:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, _value = stripped.split("=", 1)
            if name.strip() == key:
                return True
    except OSError:
        return False
    return False
