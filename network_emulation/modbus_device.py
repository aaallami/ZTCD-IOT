#!/usr/bin/env python3
"""
modbus_device.py — a real Modbus TCP server (pymodbus) standing in for a
BMS controller or Smart PDU.

Real Modbus has no built-in authentication of any kind -- this is a
well-documented, genuine limitation of the protocol, not a
simplification introduced by this test. Any host that can open a TCP
connection to the Modbus port can read and write registers. This means
the *entire* security burden for a Modbus device falls on network-layer
controls (the OVS deny-by-default rules already tested), which is
exactly the point the manuscript's architecture makes about legacy OT
protocols. This script therefore measures a real Modbus read/write
transaction completing or failing based purely on whether the OVS
flow rules let the connection through -- there is no separate
application-layer credential to bypass, because none exists in the
real protocol.
"""
import sys
import time
from pymodbus.datastore import (
    ModbusSequentialDataBlock,
    ModbusDeviceContext,
    ModbusServerContext,
)
from pymodbus.server import StartTcpServer


def build_context():
    control_block = ModbusSequentialDataBlock(1, [0] * 10)   # unit 1: control regs
    dev1 = ModbusDeviceContext(hr=control_block)
    context = ModbusServerContext(devices={1: dev1}, single=False)
    return context


if __name__ == '__main__':
    port = int(sys.argv[1])
    duration = float(sys.argv[2])
    context = build_context()

    import threading
    t = threading.Thread(target=StartTcpServer, kwargs={
        'context': context,
        'address': ('0.0.0.0', port),
    }, daemon=True)
    t.start()
    time.sleep(duration)
