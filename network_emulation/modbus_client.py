#!/usr/bin/env python3
"""modbus_client.py — attempt a real Modbus TCP write to a target
device's control register (e.g. setting a PDU outlet or BMS setpoint)."""
import sys
from pymodbus.client import ModbusTcpClient


def attempt_write(ip, port, value, timeout=1.0):
    client = ModbusTcpClient(ip, port=port, timeout=timeout)
    try:
        connected = client.connect()
        if not connected:
            return False, 'tcp-connect-failed'
        result = client.write_register(address=1, value=value, device_id=1)
        if result.isError():
            return False, f'modbus-error:{result}'
        # confirm by reading it back
        read = client.read_holding_registers(address=1, count=1, device_id=1)
        client.close()
        if read.isError():
            return True, 'write-ok-readback-failed'
        return True, f'write-ok-value-read-back={read.registers[0]}'
    except Exception as e:
        return False, f'exception:{e}'


if __name__ == '__main__':
    ip = sys.argv[1]
    port = int(sys.argv[2])
    value = int(sys.argv[3])
    ok, detail = attempt_write(ip, port, value)
    print(f"success={ok}\tdetail={detail}")
