# Minimal Windows stub for the Unix-only 'resource' module used by SWE-bench
# Provides no-op setrlimit and RLIMIT_NOFILE constant.

RLIMIT_NOFILE = 7

def setrlimit(resource, limits):
    # No-op on Windows. Accepts (int, (soft, hard)).
    return None
