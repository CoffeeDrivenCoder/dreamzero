"""Client for communicating with a policy server.

Adapted from https://github.com/robo-arena/roboarena/

"""

import logging
import inspect
import time
from typing import Dict, Tuple

import websockets.sync.client
from typing_extensions import override

from openpi_client.base_policy import BasePolicy
from openpi_client import msgpack_numpy

# The websocket server runs synchronous inference and may block for a long time
# during first-step compilation or large batched decode. Disable automatic client
# keepalive pings so long-running requests aren't misclassified as dead connections.
PING_INTERVAL_SECS = None
PING_TIMEOUT_SECS = None


def _connect(uri: str) -> websockets.sync.client.ClientConnection:
    kwargs = {
        "compression": None,
        "max_size": None,
        "ping_interval": PING_INTERVAL_SECS,
        "ping_timeout": PING_TIMEOUT_SECS,
    }
    try:
        supported = inspect.signature(websockets.sync.client.connect).parameters
        kwargs = {key: value for key, value in kwargs.items() if key in supported}
    except (TypeError, ValueError):
        pass

    try:
        return websockets.sync.client.connect(uri, **kwargs)
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        logging.warning("websockets.connect does not support one of %s; retrying with legacy kwargs.", sorted(kwargs))
        legacy_kwargs = {key: value for key, value in kwargs.items() if key in {"compression", "max_size"}}
        try:
            return websockets.sync.client.connect(uri, **legacy_kwargs)
        except TypeError:
            return websockets.sync.client.connect(uri)


def _payload_summary(obs: Dict) -> Tuple[int, str]:
    total_bytes = 0
    parts = []
    for key, value in obs.items():
        if isinstance(value, dict) and value.get("__dreamzero_image_encoding__") == "jpeg_sequence":
            nbytes = sum(len(frame) for frame in value.get("frames", []))
            total_bytes += int(nbytes)
            parts.append(
                f"{key}:encoding=jpeg_sequence shape={tuple(value.get('shape', ()))} "
                f"quality={value.get('quality')} bytes={int(nbytes)}"
            )
            continue
        nbytes = getattr(value, "nbytes", None)
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if nbytes is None:
            continue
        total_bytes += int(nbytes)
        parts.append(f"{key}:shape={tuple(shape) if shape is not None else '?'} dtype={dtype} bytes={int(nbytes)}")
    return total_bytes, "; ".join(parts)


class WebsocketClientPolicy(BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        try:
            conn = _connect(self._uri)
            metadata = msgpack_numpy.unpackb(conn.recv())
            return conn, metadata
        except Exception as exc:
            logging.info("Connection to server with ws:// failed (%s). Trying wss:// ...", exc)
            
        self._uri = "wss://" + self._uri.split("//")[1]
        conn = _connect(self._uri)
        metadata = msgpack_numpy.unpackb(conn.recv())
        return conn, metadata

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        # Notify server that we're calling the infer endpoint (as opposed to the reset endpoint)
        obs["endpoint"] = "infer"

        total_started_at = time.perf_counter()
        payload_bytes, payload_parts = _payload_summary(obs)

        pack_started_at = time.perf_counter()
        data = self._packer.pack(obs)
        packed_at = time.perf_counter()

        send_started_at = time.perf_counter()
        self._ws.send(data)
        sent_at = time.perf_counter()

        recv_started_at = time.perf_counter()
        response = self._ws.recv()
        received_at = time.perf_counter()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")

        unpack_started_at = time.perf_counter()
        result = msgpack_numpy.unpackb(response)
        unpacked_at = time.perf_counter()

        logging.info(
            "Websocket infer timing | total=%.3fs pack=%.3fs send=%.3fs recv_wait=%.3fs unpack=%.3fs "
            "payload_arrays=%.2fMB packed=%.2fMB response=%.2fMB uri=%s",
            unpacked_at - total_started_at,
            packed_at - pack_started_at,
            sent_at - send_started_at,
            received_at - recv_started_at,
            unpacked_at - unpack_started_at,
            payload_bytes / (1024 * 1024),
            len(data) / (1024 * 1024),
            len(response) / (1024 * 1024),
            self._uri,
        )
        if payload_parts:
            logging.info("Websocket infer payload detail | %s", payload_parts)
        return result

    @override
    def reset(self, reset_info: Dict) -> None:
        # Notify server that we're calling the reset endpoint (as opposed to the infer endpoint)
        reset_info["endpoint"] = "reset"

        data = self._packer.pack(reset_info)
        self._ws.send(data)
        response = self._ws.recv()
        return response

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = WebsocketClientPolicy()
    actions = client.infer({})
    print(f"Actions received: {actions}")
    client.reset({})
