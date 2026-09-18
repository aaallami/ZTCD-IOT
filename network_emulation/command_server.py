#!/usr/bin/env python3
"""
command_server.py — a small real TCP service standing in for a
Critical-zone device's control interface (BMS setpoint change, PDU
power-state toggle, UPS configuration write).

Two modes, matching the manuscript's own narrative:

  --mode legacy   : no device-identity check at all (accepts any
                    connection and executes the command). This models
                    a traditional VLAN-segmented deployment, where
                    the manuscript explicitly states device-level
                    identity enforcement is absent.

  --mode ztdc     : requires a pre-shared token (standing in for the
                    HICMS-issued, hardware-bound credential) before
                    executing the command. Any connection without the
                    correct token is logged and rejected.

This does not model privilege escalation or multi-hop chains -- it is
a single-hop authentication-bypass test, run over the real reachability
substrate already enforced by the OVS flow rules in ztdc_topo.py.
"""
import socket
import sys
import threading
import time

VALID_TOKEN = "HICMS-7f3a9c-DEVICE-CERT"


def handle_client(conn, addr, mode, logfile):
    try:
        conn.settimeout(1.0)
        data = conn.recv(256).decode(errors='replace').strip()
        src_ip = addr[0]
        if mode == 'legacy':
            conn.sendall(b"OK: command executed (no identity check)\n")
            entry = (src_ip, data, True, 'no-auth-required')
        else:  # ztdc
            if data.startswith(VALID_TOKEN):
                conn.sendall(b"OK: command executed (identity verified)\n")
                entry = (src_ip, data, True, 'valid-token')
            else:
                conn.sendall(b"DENIED: authentication failed\n")
                entry = (src_ip, data, False, 'invalid-or-missing-token')
    except Exception as e:
        entry = (addr[0], None, False, f'error:{e}')
    finally:
        conn.close()
    with open(logfile, 'a') as f:
        f.write(f"{entry[0]}\t{entry[1]}\t{entry[2]}\t{entry[3]}\n")


def run_server(port, mode, duration, logfile):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('0.0.0.0', port))
    s.listen(20)
    s.settimeout(0.5)
    end = time.time() + duration
    while time.time() < end:
        try:
            conn, addr = s.accept()
            handle_client(conn, addr, mode, logfile)
        except socket.timeout:
            continue
    s.close()


if __name__ == '__main__':
    port = int(sys.argv[1])
    mode = sys.argv[2]
    duration = float(sys.argv[3])
    logfile = sys.argv[4]
    open(logfile, 'w').close()  # truncate/create
    run_server(port, mode, duration, logfile)
