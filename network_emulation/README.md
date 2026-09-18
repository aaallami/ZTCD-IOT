# Network-emulation validation

Code for the network-emulation validation reported in Section V-G of the
manuscript: a Mininet + Open vSwitch topology enforcing (a) a VLAN-style
baseline where every host in a zone can reach every other host, and (b)
the ZTDC-IoT deny-by-default policy, tested against real, unmodified
implementations of MQTT, Modbus TCP, and a token-based command protocol
standing in for the RFID/BACnet access-control flow.

This is not hardware-in-the-loop testing: no physical device is
involved. It is real switch software (Open vSwitch) and real Linux
networking enforcing real packet-forwarding decisions, at a scale
(18 hosts) bounded by running many Linux network namespaces under one
kernel scheduler on a single host, not by the mechanism under test.

## Contents

| File | What it does |
|---|---|
| `ztdc_topo.py` | Builds the topology, installs baseline/ZTDC-IoT OVS flow rules, runs the reachability, attack-path, and real-protocol (MQTT, Modbus) tests |
| `command_server.py` / `command_client.py` | Minimal token-authenticated command service used for the attack-path test against Critical-zone devices |
| `modbus_device.py` / `modbus_client.py` | pymodbus-based Modbus TCP server/client (Modbus has no native authentication, so this isolates the network layer as the only available defense) |
| `mqtt_config/mosquitto.conf`, `mqtt_config/acl.conf` | Mosquitto broker config: password-file auth plus per-topic ACLs |

## Setup

Requires Mininet, Open vSwitch, and root privileges (standard for
network-namespace emulation):

```
sudo apt-get install mininet openvswitch-switch mosquitto mosquitto-clients
pip install pymodbus
```

Generate the Mosquitto password file referenced by `mqtt_config/mosquitto.conf`
(only `sensor01` is exercised by `ztdc_topo.py`; add further users the same
way if extending the test):

```
mosquitto_passwd -b -c mqtt_config/passwd sensor01 HICMS-cert-abc123
```

Run from this directory, so the relative paths in `mosquitto.conf` and
in `ztdc_topo.py`'s subprocess calls resolve correctly:

```
sudo python3 ztdc_topo.py
```

## Output

Prints reachability, attack-path, and real-protocol results for both
policies to stdout and writes a summary to `results.txt`. The two
protocols with no native authentication (Modbus, BACnet in the paper's
broader protocol set) show unauthorized access blocked only once the
ZTDC-IoT policy is applied; MQTT's own ACL enforcement blocks the
unauthorized topic write under both policies, since it does not depend
on network-layer segmentation.
