# Portable helpers shared by the scripts (source this file; do not execute it).
#
# tmo SECONDS CMD [ARGS...]
#   Run CMD with a deadline and exit 124 when it is exceeded, exactly like GNU
#   `timeout`. Stock macOS has no `timeout`: fall back to Homebrew coreutils'
#   `gtimeout`, and finally to a small python3 implementation (stdin, stdout and
#   stderr are inherited, so redirections and pipes work as with the real tool).
#   The fallback follows GNU `timeout` where the scripts depend on it: the command
#   runs in its own process group and the whole group is signalled at the deadline
#   (SIGTERM, then SIGKILL after 1 s), so a grandchild that holds a pipe open cannot
#   keep `$(tmo ...)` blocked; a command that cannot be started is one line on stderr
#   and status 127 (not found) or 126 (not executable); SIGINT/SIGTERM delivered to
#   the wrapper are forwarded to the group.
# TMO_IMPL names the implementation in use: timeout, gtimeout or python.
# STOCKCRAB_PY_TIMEOUT=1 forces the python3 fallback.
if [ -z "${STOCKCRAB_PY_TIMEOUT:-}" ] && command -v timeout >/dev/null 2>&1; then
  TMO_IMPL=timeout
  tmo() { timeout "$@"; }
elif [ -z "${STOCKCRAB_PY_TIMEOUT:-}" ] && command -v gtimeout >/dev/null 2>&1; then
  TMO_IMPL=gtimeout
  tmo() { gtimeout "$@"; }
elif command -v python3 >/dev/null 2>&1; then
  TMO_IMPL=python
  tmo() {
    python3 -c '
import os, signal, subprocess, sys
secs = float(sys.argv[1])
cmd = sys.argv[2:]
try:
    p = subprocess.Popen(cmd, start_new_session=True)
except FileNotFoundError:
    sys.stderr.write("tmo: failed to run command \x27%s\x27: No such file or directory\n" % cmd[0])
    sys.exit(127)
except OSError as e:
    sys.stderr.write("tmo: failed to run command \x27%s\x27: %s\n" % (cmd[0], e.strerror))
    sys.exit(126)

def kill_group(sig):
    try:
        os.killpg(p.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass

def stop(status):
    kill_group(signal.SIGTERM)
    try:
        p.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    kill_group(signal.SIGKILL)
    p.wait()
    sys.exit(status)

class Signalled(Exception):
    pass

def on_signal(signum, frame):
    # The command is outside the foreground process group: pass the signal on.
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, signal.SIG_IGN)
    raise Signalled(signum)

for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(s, on_signal)
try:
    rc = p.wait(timeout=secs)
except subprocess.TimeoutExpired:
    stop(124)
except Signalled as e:
    stop(128 + e.args[0])
sys.exit(rc if rc >= 0 else 128 - rc)
' "$@"
  }
else
  echo "need one of: timeout, gtimeout, python3" >&2
  exit 2
fi
