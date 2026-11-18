#!/usr/bin/env python3
"""Make a Windows source tree buildable on a case-sensitive file system: for every #include that only resolves
case-insensitively, add a symlink with the spelling the source uses. usage: casefix.py <root> <include dir>..."""
import os, re, sys
root, incs = os.path.abspath(sys.argv[1]), [os.path.abspath(i) for i in sys.argv[2:]]
index = {}
for d, _, files in os.walk(root):
    for f in files:
        index.setdefault(os.path.join(d, f).lower(), os.path.join(d, f))
made = 0
for d, _, files in os.walk(root):
    for f in files:
        if not f.lower().endswith(('.cpp', '.h', '.hpp', '.c', '.inl', '.rc')):
            continue
        p = os.path.join(d, f)
        try:
            text = open(p, encoding='utf-8', errors='ignore').read()
        except OSError:
            continue
        for inc in re.findall(r'#\s*include\s*[<"]([^>"]+)[>"]', text):
            inc = inc.replace('\\', '/')
            for base in [d] + incs:
                want = os.path.normpath(os.path.join(base, inc))
                if os.path.exists(want):
                    break
                real = index.get(want.lower())
                if real:
                    os.makedirs(os.path.dirname(want), exist_ok=True)
                    os.symlink(os.path.relpath(real, os.path.dirname(want)), want)
                    made += 1
                    break
print(f'{made} case links')
