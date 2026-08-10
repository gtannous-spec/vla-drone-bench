"""
sync_rpc.py — Synchronous msgpack-RPC client for AirSim.

Drop-in replacement for msgpackrpc.Client and msgpackrpc.Address that uses
blocking sockets instead of Tornado's IOLoop. This bypasses the Tornado 6 +
asyncio incompatibility that causes deadlocks on Python 3.10+.

AirSim's RPC protocol:
    Request:  [0, msgid, method, params]  (msgpack-encoded)
    Response: [1, msgid, error,  result]  (msgpack-encoded)
"""

import socket
import threading

import msgpack


class SyncAddress:
    """Drop-in replacement for msgpackrpc.Address."""

    def __init__(self, host, port, family=socket.AF_UNSPEC):
        self._host = host
        self._port = port

    @property
    def host(self):
        return self._host

    @property
    def port(self):
        return self._port

    def unpack(self):
        return (self._host, self._port)


class SyncFuture:
    """Drop-in replacement for msgpackrpc.future.Future.

    Since our RPC calls are synchronous, the result is available immediately
    after the call returns. join()/get() just return the stored value.
    """

    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    def get(self):
        if self._error is not None:
            from msgpackrpc.error import RPCError
            raise RPCError(self._error)
        return self._result

    def join(self):
        self.get()


class SyncClient:
    """Drop-in replacement for msgpackrpc.Client using blocking sockets.

    Maintains a persistent TCP connection to the AirSim RPC server.
    Thread-safe via a lock around send/recv.
    """

    def __init__(self, address, timeout=3600, loop=None, builder=None,
                 reconnect_limit=5, pack_encoding='utf-8', unpack_encoding=None):
        self._address = address
        self._timeout = timeout
        self._reconnect_limit = reconnect_limit
        self._sock = None
        self._msgid = 0
        self._lock = threading.Lock()
        self._unpacker = msgpack.Unpacker(raw=False)
        self._default = lambda obj: obj.to_msgpack() if hasattr(obj, 'to_msgpack') else obj

    def _connect(self):
        if self._sock is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self._timeout)
        sock.connect(self._address.unpack())
        self._sock = sock

    def _ensure_connected(self):
        for attempt in range(self._reconnect_limit):
            try:
                self._connect()
                return
            except (OSError, ConnectionError):
                self._sock = None
                if attempt == self._reconnect_limit - 1:
                    from msgpackrpc.error import TransportError
                    raise TransportError("Retry connection over the limit")
                import time
                time.sleep(0.5)

    def call(self, method, *args):
        """Send an RPC request and block until the response arrives."""
        with self._lock:
            self._ensure_connected()
            self._msgid += 1
            msgid = self._msgid

            request = msgpack.packb([0, msgid, method, args], default=self._default, use_bin_type=True)
            try:
                self._sock.sendall(request)
            except (OSError, BrokenPipeError):
                self._sock = None
                self._ensure_connected()
                self._sock.sendall(request)

            return self._recv_response(msgid)

    def call_async(self, method, *args):
        """Synchronous call wrapped in a Future for API compat with airsim."""
        try:
            result = self.call(method, *args)
            return SyncFuture(result=result)
        except Exception as e:
            return SyncFuture(error=str(e))

    def _recv_response(self, expected_msgid):
        """Read and decode the msgpack-RPC response."""
        while True:
            try:
                chunk = self._sock.recv(65536)
            except socket.timeout:
                from msgpackrpc.error import TimeoutError
                raise TimeoutError("Request timed out")
            if not chunk:
                from msgpackrpc.error import TransportError
                self._sock = None
                raise TransportError("Connection closed by server")

            self._unpacker.feed(chunk)
            for message in self._unpacker:
                if not isinstance(message, (list, tuple)) or len(message) != 4:
                    continue
                msg_type, msg_id, error, result = message
                if msg_type == 1 and msg_id == expected_msgid:
                    if error is not None:
                        from msgpackrpc.error import RPCError
                        raise RPCError(error)
                    return result

    def close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
