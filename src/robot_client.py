# Robot communication
import socket
import time
from typing import Optional


class RobotClient:
    def __init__(self, host: str, port: int, reconnect_delay_sec: float) -> None:
        self.host = host
        self.port = port
        self.reconnect_delay_sec = reconnect_delay_sec
        self.sock: Optional[socket.socket] = None

    def connect(self) -> None:
        while self.sock is None:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2.0)
                sock.connect((self.host, self.port))
                sock.settimeout(None)
                self.sock = sock
                print(f"Connected to robot: {self.host}:{self.port}")
            except OSError as exc:
                print(f"Connect failed ({exc}); retrying in {self.reconnect_delay_sec:.1f}s")
                time.sleep(self.reconnect_delay_sec)

    def send_char(self, command: str) -> None:
        if len(command) != 1:
            raise ValueError("Command must be a single character")

        if self.sock is None:
            self.connect()

        try:
            assert self.sock is not None
            self.sock.send(bytes([ord(command)]))
        except OSError as exc:
            print(f"Send failed ({exc}); reconnecting...")
            self.close()
            self.connect()

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None