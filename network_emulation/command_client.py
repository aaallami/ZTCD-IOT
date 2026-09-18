#!/usr/bin/env python3
"""command_client.py — attempt one command against a target host's
command_server, either with the correct token (legitimate device) or
without it (attacker / unprovisioned device)."""
import socket
import sys

VALID_TOKEN = "HICMS-7f3a9c-DEVICE-CERT"


def attempt(target_ip, port, use_valid_token, timeout=0.3):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((target_ip, port))
        payload = f"{VALID_TOKEN} SET_STATE=1" if use_valid_token else "SET_STATE=1"
        s.sendall(payload.encode())
        resp = s.recv(256).decode(errors='replace').strip()
        s.close()
        return True, resp
    except Exception as e:
        return False, str(e)


if __name__ == '__main__':
    target_ip = sys.argv[1]
    port = int(sys.argv[2])
    use_valid = sys.argv[3] == '1'
    reached, resp = attempt(target_ip, port, use_valid)
    print(f"reached={reached}\tresponse={resp}")
