"""Check CARLA's TCP ports before starting the Unreal game process (stdlib only)."""

import errno
import os
import shutil
import socket
import subprocess
import sys


PORT_FLAGS = {
    "-world-port": "rpc", "-carla-port": "rpc", "-carla-rpc-port": "rpc",
    "-carla-world-port": "rpc", "-carla-streaming-port": "streaming",
    "-carla-secondary-port": "secondary",
}


def launch_command(editor, project, arguments):
    """Normalize port overrides so the checked ports equal Unreal's arguments."""
    selected, remaining = {}, []
    for argument in arguments:
        flag, separator, value = argument.partition("=")
        kind = PORT_FLAGS.get(flag.lower())
        if kind is None:
            remaining.append(argument)
            continue
        if not separator or not value.isascii() or not value.isdecimal():
            raise ValueError(f"Use {flag}=PORT with a numeric TCP port")
        port = int(value)
        if kind in selected and selected[kind] != port:
            raise ValueError(f"Conflicting {kind} port overrides")
        selected[kind] = port
    rpc = selected.get("rpc", 2000)
    ports = {"rpc": rpc, "streaming": selected.get("streaming", rpc + 1),
             "secondary": selected.get("secondary", rpc + 2)}
    if any(not 1 <= port <= 65535 for port in ports.values()):
        raise ValueError("CARLA ports must be between 1 and 65535 (default companion ports are RPC+1 and RPC+2)")
    if len(set(ports.values())) != 3:
        raise ValueError("RPC, streaming and secondary ports must differ")
    command = [str(editor), str(project), "/Game/Carla/Maps/Town01", "-game", "-vulkan",
               "-RenderOffScreen", f"-carla-rpc-port={ports['rpc']}",
               f"-carla-streaming-port={ports['streaming']}",
               f"-carla-secondary-port={ports['secondary']}", "-quality-level=Epic",
               "-carla-rgb-only", "-nosound", "-unattended", "-log", *remaining]
    return command, ports


def check_ports(ports):
    """Bind, without connecting/ticking/resetting an existing CARLA session."""
    sockets, occupied = [], []
    try:
        for kind, port in ports.items():
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sockets.append(listener)
            # Match CARLA's reuse-address behavior while rejecting active listeners.
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                listener.bind(("0.0.0.0", port))
            except OSError as exc:
                if exc.errno != errno.EADDRINUSE:
                    raise OSError(f"Cannot check CARLA {kind} port {port}: {exc}") from exc
                occupied.append((kind, port))
    finally:
        for listener in sockets:
            listener.close()
    if occupied:
        names = ", ".join(f"{kind}={port}" for kind, port in occupied)
        raise RuntimeError(f"CARLA ports already in use: {names}.\n"
                           "An existing server may still be running or stuck shutting down. "
                           "Refusing to start another Unreal process, which would crash with "
                           "'bind: Address already in use'.\n"
                           "Use the existing server if it responds, or stop its owning process before restarting.\n"
                           "For a separate server, choose an unused range, e.g. -carla-rpc-port=2010 "
                           "(also uses 2011 and 2012), and update your client configuration.")


def port_owners(ports):
    if not shutil.which("ss"):
        return ""
    query = " or ".join(f"sport = :{port}" for port in ports.values())
    try:
        result = subprocess.run(["ss", "-ltnp", query], capture_output=True, text=True, timeout=3)
        return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) < 2:
        print("Usage: server_preflight.py EDITOR PROJECT [UNREAL_FLAGS...]", file=sys.stderr)
        return 2
    ports = None
    try:
        command, ports = launch_command(args[0], args[1], args[2:])
        check_ports(ports)
    except (ValueError, RuntimeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        if ports:
            owners = port_owners(ports)
            if owners:
                print(owners, file=sys.stderr)
        return 2
    os.execv(command[0], command)


if __name__ == "__main__":
    sys.exit(main())
