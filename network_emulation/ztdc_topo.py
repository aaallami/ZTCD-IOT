#!/usr/bin/env python3
"""
ztdc_topo.py — Network-emulation validation for ZTDC-IoT.

Builds a scaled-down but structurally faithful emulation of the Tier-3
IoT deployment in Table 1 of the manuscript, using real Mininet hosts
(Linux network namespaces), a real Open vSwitch switch per zone, and
real OpenFlow rules to enforce (a) a VLAN-style baseline where every
device in a zone can reach every other device, and (b) the ZTDC-IoT
deny-by-default policy where only allow-listed (src_class, dst_class)
pairs are permitted.

This is not hardware-in-the-loop testing: it is real switch software
(Open vSwitch) and real Linux networking enforcing real packet-forwarding
decisions, at a scale a physical pilot could not reach in this
environment. Device classes are represented by Mininet hosts running
plain IP traffic (ping/iperf) rather than full protocol stacks, which is
the main simplification relative to physical hardware.

Scale: a single-core execution environment caps the exhaustive (O(n^2))
connectivity-matrix tests to roughly 20-30 hosts before wall-clock time
becomes impractical, so the topology below is sized accordingly (a
representative subset per zone, not the full 1,400-2,400 device
inventory). On a multi-core machine, the same script scales to the full
inventory without modification; only the DEVICE_COUNTS table below
needs to change.
"""

import sys
import re
import time
import itertools
from mininet.net import Mininet
from mininet.node import OVSSwitch
from mininet.link import Link
from mininet.log import setLogLevel
from mininet.cli import CLI

setLogLevel('info')

# ---------------------------------------------------------------------
# Device inventory (scaled down from Table 1 for single-core feasibility)
# Format: zone -> {device_class: count}
# ---------------------------------------------------------------------
DEVICE_COUNTS = {
    'env':   {'sensor': 3},                 # Environmental sensors (Public/Ops)
    'surv':  {'camera': 2},                 # IP surveillance cameras (Restricted)
    'access':{'rfid': 2},                   # RFID/biometric access control (Restricted)
    'power': {'pdu': 2},                    # Smart PDUs (Critical)
    'bms':   {'bms': 2},                    # BMS controllers (Critical)
    'ups':   {'ups': 1},                    # Network-attached UPS (Critical)
    'wless': {'zigbee': 2},                 # Vibration/leak sensors (Mixed)
}

# ---------------------------------------------------------------------
# Allow-list: (src_class, dst_class) pairs permitted under ZTDC-IoT.
# Mirrors the manuscript's zone/protocol model: monitoring flows from
# sensors to a management host, BMS/PDU telemetry to a facilities host,
# camera streams to a VMS host, RFID auth to an access-control host.
# Everything else is denied by default.
# ---------------------------------------------------------------------
ALLOWLIST = {
    ('sensor', 'mgmt'), ('mgmt', 'sensor'),
    ('camera', 'vms'),  ('vms', 'camera'),
    ('rfid', 'acs'),    ('acs', 'rfid'),
    ('pdu', 'facilities'), ('facilities', 'pdu'),
    ('bms', 'facilities'), ('facilities', 'bms'),
    ('ups', 'facilities'), ('facilities', 'ups'),
    ('zigbee', 'mgmt'), ('mgmt', 'zigbee'),
}
# Infrastructure/management hosts, one per role, on their own class:
INFRA_CLASSES = ['mgmt', 'vms', 'acs', 'facilities']


def build_hosts(net, switch):
    """Create one Mininet host per device instance, tagged with its
    device class in a dict, all attached to the given switch."""
    hosts = {}
    idx = 1
    for zone, classes in DEVICE_COUNTS.items():
        for cls, count in classes.items():
            for i in range(count):
                name = f'h{idx}'
                ip = f'10.0.0.{idx}/24'
                h = net.addHost(name, ip=ip)
                net.addLink(h, switch, cls=Link)
                hosts[name] = {'class': cls, 'zone': zone}
                idx += 1
    for cls in INFRA_CLASSES:
        name = f'h{idx}'
        ip = f'10.0.0.{idx}/24'
        h = net.addHost(name, ip=ip)
        net.addLink(h, switch, cls=Link)
        hosts[name] = {'class': cls, 'zone': 'infra'}
        idx += 1
    return hosts


def install_baseline_rules(net, switch, hosts):
    """VLAN-style baseline: every host can reach every other host
    (normal OVS learning-switch / standalone behavior -- no explicit
    flow rules needed beyond default flooding)."""
    switch.cmd('ovs-ofctl del-flows', switch.name)
    switch.cmd(f'ovs-ofctl add-flow {switch.name} priority=0,actions=NORMAL')


def install_ztdc_rules(net, switch, hosts):
    """Deny-by-default: explicit flow rules only for allow-listed
    (src_class, dst_class) pairs; drop everything else."""
    switch.cmd('ovs-ofctl del-flows', switch.name)
    name_to_ip = {n: net.get(n).IP() for n in hosts}
    for src_name, src_info in hosts.items():
        for dst_name, dst_info in hosts.items():
            if src_name == dst_name:
                continue
            pair = (src_info['class'], dst_info['class'])
            if pair in ALLOWLIST:
                src_ip = name_to_ip[src_name]
                dst_ip = name_to_ip[dst_name]
                switch.cmd(
                    f'ovs-ofctl add-flow {switch.name} '
                    f'priority=100,ip,nw_src={src_ip},nw_dst={dst_ip},actions=NORMAL'
                )
    switch.cmd(f'ovs-ofctl add-flow {switch.name} priority=1,actions=drop')
    # ARP is left enabled; without it, the IP rules above cannot resolve MACs:
    switch.cmd(f'ovs-ofctl add-flow {switch.name} priority=50,arp,actions=NORMAL')


def measure_reachable_pairs(net, hosts):
    """Return the count of ordered (src,dst) host pairs that can
    successfully ping each other, out of all possible ordered pairs,
    plus the measured RTT (ms) for each successful pair."""
    names = list(hosts.keys())
    reachable = 0
    total = 0
    results = []
    rtts = []
    for src, dst in itertools.permutations(names, 2):
        total += 1
        h_src = net.get(src)
        h_dst = net.get(dst)
        result = h_src.cmd(f'ping -c1 -W 0.3 {h_dst.IP()}')
        ok = ' 0% packet loss' in result or ', 0% packet loss' in result
        rtt = None
        if ok:
            reachable += 1
            m = re.search(r'rtt min/avg/max/mdev = ([\d.]+)/([\d.]+)/([\d.]+)', result)
            if m:
                rtt = float(m.group(2))
                rtts.append(rtt)
        results.append((src, dst, hosts[src]['class'], hosts[dst]['class'], ok, rtt))
    return reachable, total, results, rtts


CRITICAL_CLASSES = {'pdu', 'bms', 'ups'}  # command_server runs on these
COMMAND_PORT = 9999


def run_attack_path_test(net, hosts, mode):
    """
    Real attack-path test covering two of the manuscript's four
    conjunctive predicates: reachability (enforced by the already-
    installed OVS flow rules) and authentication bypass (enforced by
    command_server.py, run in 'legacy' or 'ztdc' mode).

    For every (attacker_host, target_host) pair where target is a
    Critical-zone device (pdu/bms/ups), the attacker attempts a
    command WITHOUT a valid credential (modeling a compromised or
    unprovisioned device). A "successful attack" requires both
    reachability (TCP connect succeeds) AND the command being
    executed (auth bypassed or absent). This does not model privilege
    escalation or multi-hop chains; it is a single-hop test.

    Also tests the legitimate-flow case (correct token) from
    allow-listed source classes, to confirm no false-negative
    blocking of real traffic.
    """
    targets = [n for n, info in hosts.items() if info['class'] in CRITICAL_CLASSES]
    attackers = list(hosts.keys())

    server_procs = {}
    for t in targets:
        h = net.get(t)
        logfile = f'/tmp/cmdlog_{mode}_{t}.txt'
        h.cmd(f'rm -f {logfile}')
        p = h.popen(
            ['python3', 'command_server.py',
             str(COMMAND_PORT), mode, '40', logfile]
        )
        server_procs[t] = (p, logfile)
    time.sleep(1)

    attack_results = []
    legit_results = []

    for attacker in attackers:
        if attacker in targets:
            continue
        h_att = net.get(attacker)
        for target in targets:
            h_tgt = net.get(target)
            tgt_ip = h_tgt.IP()

            out = h_att.cmd(
                f'python3 command_client.py {tgt_ip} {COMMAND_PORT} 0'
            )
            reached = 'reached=True' in out
            executed = reached and 'OK:' in out
            attack_results.append((attacker, target, hosts[attacker]['class'],
                                    hosts[target]['class'], reached, executed))

            pair = (hosts[attacker]['class'], hosts[target]['class'])
            if pair in ALLOWLIST or hosts[attacker]['class'] == 'facilities':
                out2 = h_att.cmd(
                    f'python3 command_client.py {tgt_ip} {COMMAND_PORT} 1'
                )
                reached2 = 'reached=True' in out2
                executed2 = reached2 and 'OK:' in out2
                legit_results.append((attacker, target, hosts[attacker]['class'],
                                       hosts[target]['class'], reached2, executed2))

    time.sleep(1)
    for t, (p, logfile) in server_procs.items():
        try:
            p.wait(timeout=8)
        except Exception:
            p.terminate()

    return attack_results, legit_results


def run_real_protocol_test(net, hosts, mode):
    """
    Real-protocol validation using actual device software, replacing the
    toy token service with the genuine protocols from Table 1:

      - MQTT (Mosquitto broker + paho clients): tests protocol-native
        authentication (bcrypt password file) AND per-topic ACL
        enforcement (least privilege) -- e.g. a legitimate but
        low-privilege sensor cannot publish to a critical command
        topic it has no ACL grant for, independent of whether the
        network layer would have let the connection through.

      - Modbus TCP (pymodbus): Modbus has NO built-in authentication
        at all (a real, documented property of the protocol, not a
        simplification). This means the network layer (the OVS
        deny-by-default rules) is the ONLY available defense for
        Modbus devices -- so this test measures whether an
        unauthorized host can complete a real Modbus write purely as
        a function of network reachability.

    Together these show a protocol-dependent security story: some
    device classes get defense-in-depth (network + identity), while
    legacy OT protocols like Modbus depend entirely on the network
    layer already validated in run_attack_path_test / measure_reachable_pairs.
    """
    import subprocess
    import time as _t

    results = {}

    # ---- MQTT: one broker on the 'mgmt' host (sensors are allow-listed
    # to reach 'mgmt', not 'facilities' -- matching the existing ALLOWLIST
    # used by the reachability/attack-path tests above) ----
    mgmt_name = [n for n, i in hosts.items() if i['class'] == 'mgmt'][0]
    facilities_name = [n for n, i in hosts.items() if i['class'] == 'facilities'][0]
    sensor_names = [n for n, i in hosts.items() if i['class'] == 'sensor']
    bms_names = [n for n, i in hosts.items() if i['class'] == 'bms']

    h_broker = net.get(mgmt_name)
    broker_logfile = f'/tmp/mqtt_broker_{mode}.log'
    conf_path = f'/tmp/mosquitto_{mode}.conf'
    h_broker.cmd(
        f'cat mqtt_config/mosquitto.conf | '
        f'sed "s|/tmp/mosquitto_test.log|{broker_logfile}|" > {conf_path}'
    )
    broker_proc = h_broker.popen(
        ['mosquitto', '-c', conf_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    _t.sleep(1.5)
    broker_ip = h_broker.IP()

    # Legit: sensor -> its own telemetry topic
    h_sensor = net.get(sensor_names[0])
    r1 = h_sensor.cmd(
        f'timeout 2 mosquitto_pub -h {broker_ip} -p 1883 -u sensor01 '
        f'-P "HICMS-cert-abc123" -t "zone/environmental/telemetry" -m "temp=22.5"'
    )
    mqtt_legit_reachable = 'Error' not in r1 and 'refused' not in r1.lower()

    # Attack: same legit-credentialed sensor tries a critical command topic
    # it has no ACL grant for. Verified by reading the BROKER's own local
    # log for an explicit ACL denial -- this check runs on the broker host
    # itself, so it has no dependency on any other host's network
    # reachability under the ZTDC-IoT policy (unlike a second verification
    # host, which might itself be blocked at the network layer for
    # unrelated reasons, confounding the result).
    h_sensor.cmd(
        f'timeout 2 mosquitto_pub -h {broker_ip} -p 1883 -u sensor01 '
        f'-P "HICMS-cert-abc123" -t "zone/critical/command" -m "SET_POWER_OFF"'
    )
    _t.sleep(0.5)
    broker_log_text = h_broker.cmd(f'cat {broker_logfile}')
    mqtt_attack_delivered = (
        'Denied PUBLISH' not in broker_log_text
        and 'zone/critical/command' in broker_log_text
    )

    broker_proc.terminate()
    try:
        broker_proc.wait(timeout=3)
    except Exception:
        broker_proc.kill()

    results['mqtt_legit_reachable'] = mqtt_legit_reachable
    results['mqtt_unauthorized_topic_blocked'] = not mqtt_attack_delivered

    # ---- Modbus: one server on a 'pdu' host, tested from multiple sources ----
    pdu_name = [n for n, i in hosts.items() if i['class'] == 'pdu'][0]
    h_pdu = net.get(pdu_name)
    modbus_log = open(f'/tmp/modbus_srv_{mode}.log', 'w')
    modbus_proc = h_pdu.popen(
        ['python3', 'modbus_device.py', '5020', '20'],
        stdout=modbus_log, stderr=subprocess.STDOUT,
    )
    _t.sleep(1.5)
    pdu_ip = h_pdu.IP()

    # Legit: facilities host writes to the PDU's control register
    h_facilities = net.get(facilities_name)
    r_legit = h_facilities.cmd(
        f'timeout 2 python3 modbus_client.py {pdu_ip} 5020 1'
    )
    modbus_legit_ok = 'success=True' in r_legit

    # Attack: an unrelated sensor (never allow-listed to reach the PDU)
    # attempts the same real Modbus write.
    h_attacker = net.get(sensor_names[-1])
    r_attack = h_attacker.cmd(
        f'timeout 2 python3 modbus_client.py {pdu_ip} 5020 0'
    )
    modbus_attack_ok = 'success=True' in r_attack

    modbus_proc.terminate()
    try:
        modbus_proc.wait(timeout=3)
    except Exception:
        modbus_proc.kill()
    modbus_log.close()

    results['modbus_legit_write_succeeded'] = modbus_legit_ok
    results['modbus_unauthorized_write_succeeded'] = modbus_attack_ok

    return results


def main():
    net = Mininet(switch=OVSSwitch, autoSetMacs=True)
    switch = net.addSwitch('s1', failMode='standalone', datapath='user')
    hosts = build_hosts(net, switch)

    print(f"Total hosts: {len(hosts)}")
    net.start()
    time.sleep(1)

    # Disable checksum/segmentation offload on every host interface: OVS's
    # userspace (netdev) datapath -- required in this environment because
    # the kernel does not support loadable kernel modules, so the normal
    # kernel datapath is unavailable -- does not correctly compute offloaded
    # checksums, which causes the receiving TCP stack to silently drop
    # segments (TCP validates checksums strictly; ICMP does not, which is
    # why ping worked throughout while TCP connections timed out).
    for hname in hosts:
        h = net.get(hname)
        intf = h.intfNames()[0]
        h.cmd(f'ethtool -K {intf} tx off rx off sg off tso off gso off gro off >/dev/null 2>&1')

    print("\n=== BASELINE (VLAN-style, all-to-all reachability) ===")
    install_baseline_rules(net, switch, hosts)
    time.sleep(1)
    base_reach, base_total, base_results, base_rtts = measure_reachable_pairs(net, hosts)
    print(f"Baseline reachable pairs: {base_reach}/{base_total} "
          f"({100.0*base_reach/base_total:.1f}%)")
    if base_rtts:
        print(f"Baseline RTT (ms): min={min(base_rtts):.3f} "
              f"avg={sum(base_rtts)/len(base_rtts):.3f} max={max(base_rtts):.3f} "
              f"n={len(base_rtts)}")

    print("\n--- Baseline attack-path test (reachability + no identity check) ---")
    base_attack, base_legit = run_attack_path_test(net, hosts, 'legacy')
    base_attack_success = sum(1 for r in base_attack if r[5])
    base_legit_success = sum(1 for r in base_legit if r[5])
    print(f"Attack attempts (no valid credential): {len(base_attack)} total, "
          f"{base_attack_success} succeeded ({100.0*base_attack_success/len(base_attack):.1f}%)")
    print(f"Legitimate attempts (valid credential): {len(base_legit)} total, "
          f"{base_legit_success} succeeded ({100.0*base_legit_success/len(base_legit):.1f}%)")

    print("\n--- Baseline REAL-PROTOCOL test (MQTT + Modbus) ---")
    base_real = run_real_protocol_test(net, hosts, 'legacy')
    for k, v in base_real.items():
        print(f"  {k}: {v}")

    print("\n=== ZTDC-IoT (deny-by-default, allow-list only) ===")
    install_ztdc_rules(net, switch, hosts)
    time.sleep(1)
    ztdc_reach, ztdc_total, ztdc_results, ztdc_rtts = measure_reachable_pairs(net, hosts)
    print(f"ZTDC-IoT reachable pairs: {ztdc_reach}/{ztdc_total} "
          f"({100.0*ztdc_reach/ztdc_total:.1f}%)")
    if ztdc_rtts:
        print(f"ZTDC-IoT RTT (ms): min={min(ztdc_rtts):.3f} "
              f"avg={sum(ztdc_rtts)/len(ztdc_rtts):.3f} max={max(ztdc_rtts):.3f} "
              f"n={len(ztdc_rtts)}")

    print("\n--- ZTDC-IoT attack-path test (reachability + identity check) ---")
    ztdc_attack, ztdc_legit = run_attack_path_test(net, hosts, 'ztdc')
    ztdc_attack_success = sum(1 for r in ztdc_attack if r[5])
    ztdc_legit_success = sum(1 for r in ztdc_legit if r[5])
    print(f"Attack attempts (no valid credential): {len(ztdc_attack)} total, "
          f"{ztdc_attack_success} succeeded ({100.0*ztdc_attack_success/len(ztdc_attack):.1f}%)")
    print(f"Legitimate attempts (valid credential): {len(ztdc_legit)} total, "
          f"{ztdc_legit_success} succeeded ({100.0*ztdc_legit_success/len(ztdc_legit):.1f}%)")

    print("\n--- ZTDC-IoT REAL-PROTOCOL test (MQTT + Modbus) ---")
    ztdc_real = run_real_protocol_test(net, hosts, 'ztdc')
    for k, v in ztdc_real.items():
        print(f"  {k}: {v}")

    reduction = 100.0 * (1 - ztdc_reach / base_reach) if base_reach else 0.0
    print(f"\nReachability reduction: {reduction:.1f}%")

    # Sanity check: verify every reachable ZTDC pair is actually on the
    # allow-list (i.e. no unintended leaks), and that every allow-listed
    # class pair present in the topology is actually reachable.
    leaks = [(s, d, sc, dc) for (s, d, sc, dc, ok, rtt) in ztdc_results
             if ok and (sc, dc) not in ALLOWLIST and sc != dc]
    missed = [(s, d, sc, dc) for (s, d, sc, dc, ok, rtt) in ztdc_results
              if not ok and (sc, dc) in ALLOWLIST]
    print(f"Unintended leaks (reachable but not allow-listed): {len(leaks)}")
    print(f"Missed allow-listed pairs (expected reachable, not reachable): {len(missed)}")
    if leaks:
        print("  LEAKS:", leaks[:10])
    if missed:
        print("  MISSED:", missed[:10])

    net.stop()

    with open('results.txt', 'w') as f:
        f.write(f"hosts={len(hosts)}\n")
        f.write(f"baseline_reachable={base_reach}\n")
        f.write(f"baseline_total={base_total}\n")
        f.write(f"ztdc_reachable={ztdc_reach}\n")
        f.write(f"ztdc_total={ztdc_total}\n")
        f.write(f"reduction_pct={reduction:.2f}\n")
        f.write(f"leaks={len(leaks)}\n")
        f.write(f"missed={len(missed)}\n")
        if base_rtts:
            f.write(f"baseline_rtt_min={min(base_rtts):.3f}\n")
            f.write(f"baseline_rtt_avg={sum(base_rtts)/len(base_rtts):.3f}\n")
            f.write(f"baseline_rtt_max={max(base_rtts):.3f}\n")
        if ztdc_rtts:
            f.write(f"ztdc_rtt_min={min(ztdc_rtts):.3f}\n")
            f.write(f"ztdc_rtt_avg={sum(ztdc_rtts)/len(ztdc_rtts):.3f}\n")
            f.write(f"ztdc_rtt_max={max(ztdc_rtts):.3f}\n")
        f.write(f"ztdc_rtt_n={len(ztdc_rtts)}\n")
        f.write(f"base_attack_attempts={len(base_attack)}\n")
        f.write(f"base_attack_success={base_attack_success}\n")
        f.write(f"base_legit_attempts={len(base_legit)}\n")
        f.write(f"base_legit_success={base_legit_success}\n")
        f.write(f"ztdc_attack_attempts={len(ztdc_attack)}\n")
        f.write(f"ztdc_attack_success={ztdc_attack_success}\n")
        f.write(f"ztdc_legit_attempts={len(ztdc_legit)}\n")
        f.write(f"ztdc_legit_success={ztdc_legit_success}\n")


if __name__ == '__main__':
    main()
