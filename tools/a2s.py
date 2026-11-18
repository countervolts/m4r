"""Decode an ipsw .a2s symbol cache (Go gob map of address -> symbol name) by pattern scanning."""
import sys


def _guint(d, p):
    b = d[p]
    if b < 128:
        return b, p + 1
    n = 256 - b
    if n > 8:
        return None, p
    return int.from_bytes(d[p + 1:p + 1 + n], 'big'), p + 1 + n


def load(path, lo=0x250000000, hi=0x280000000):
    d = open(path, 'rb').read()
    out = {}
    p = 0
    while p < len(d) - 2:
        v, q = _guint(d, p)
        if v is not None and lo < v < hi:
            n, r = _guint(d, q)
            if n is not None and 0 < n < 400 and r + n <= len(d):
                s = d[r:r + n]
                if all(32 <= c < 127 for c in s):
                    out[v] = s.decode()
                    p = r + n
                    continue
        p += 1
    return out


if __name__ == '__main__':
    for k, v in sorted(load(sys.argv[1]).items()):
        print(hex(k), v)
