#!/usr/bin/env python3
"""
new_overnight.py — Quick-start script for the MBH+PBH+SA optimizer suite.

Kills existing optimizer processes, launches:
  - pbh.py (standalone, pop=8, 2000 rounds)
  - run_overnight.py (3 PBH + 3 SA workers, 18000s)
Then tails output.
"""

import os, sys, signal, subprocess, time

WORK_DIR = '/home/camcore/.openclaw/workspace/shape-packing-challenge'


def kill_existing():
    """Kill any existing optimizer processes."""
    patterns = ['pbh.py', 'mbh.py', 'run_overnight.py', 'sa_v2.py', 'fss.py']
    my_pid = os.getpid()
    killed = 0
    try:
        result = subprocess.run(['pgrep', '-f', '|'.join(patterns)],
                                capture_output=True, text=True)
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            pid = int(line.strip())
            if pid == my_pid:
                continue
            try:
                os.kill(pid, signal.SIGTERM)
                killed += 1
            except ProcessLookupError:
                pass
    except Exception:
        pass

    if killed:
        print(f"Killed {killed} existing optimizer process(es)")
        time.sleep(2)
    else:
        print("No existing optimizer processes found")


def main():
    os.chdir(WORK_DIR)

    print("=== new_overnight.py — MBH+PBH+FSS optimizer suite ===")
    print()

    # Kill existing processes
    kill_existing()

    # Launch standalone PBH
    print("Launching: pbh.py --pop 8 --rounds 2000")
    pbh_log = open('pbh_standalone.log', 'w')
    pbh_proc = subprocess.Popen(
        [sys.executable, 'pbh.py', '--pop', '8', '--rounds', '2000'],
        stdout=pbh_log, stderr=subprocess.STDOUT,
        cwd=WORK_DIR
    )
    print(f"  PID: {pbh_proc.pid}")

    # Launch run_overnight
    print("Launching: run_overnight.py --runtime 18000")
    overnight_log = open('overnight_run.log', 'w')
    overnight_proc = subprocess.Popen(
        [sys.executable, 'run_overnight.py', '--runtime', '18000'],
        stdout=overnight_log, stderr=subprocess.STDOUT,
        cwd=WORK_DIR
    )
    print(f"  PID: {overnight_proc.pid}")

    print()
    print("Tailing output (Ctrl+C to detach, processes continue in background)...")
    print("=" * 60)

    # Tail both log files
    try:
        tail_proc = subprocess.Popen(
            ['tail', '-f', 'pbh_standalone.log', 'overnight_run.log',
             'pbh_log.txt', 'mbh_log.txt'],
            cwd=WORK_DIR
        )
        tail_proc.wait()
    except KeyboardInterrupt:
        print("\nDetached. Processes running in background:")
        print(f"  pbh.py PID: {pbh_proc.pid}")
        print(f"  run_overnight.py PID: {overnight_proc.pid}")
        print("Kill with: kill {} {}".format(pbh_proc.pid, overnight_proc.pid))


if __name__ == '__main__':
    main()
