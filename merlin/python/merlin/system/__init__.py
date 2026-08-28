"""Host+device system model: what we compile for, and how the pieces are reached."""
from .model import Device, Host, Link, System  # noqa: F401
from .derive import device_for, host_from_board, link_for, system_for  # noqa: F401
