#!/usr/bin/env python3
"""Generates m4r's plans for a game from your own MetalFX files and puts them, with m4r.dll, next to OptiScaler.

    m4r/m4r.dll
    m4r/plans/bbr-<display>-<render>        upscaler plan, reversed depth (what most engines use)
    m4r/plans/bbr-<display>-<render>-fwd    upscaler plan, forward depth
    m4r/plans/fg-<display>-<render>         frame generation plan
    m4r/plans/dn-<render>                   denoiser plan (Ray Reconstruction), followed by the upscaler
    m4r/plans/dn3-<display>-<render>        Ray Reconstruction with MetalFX's own upscaling network (preferred when present)

A plan exists per display and render resolution pair. The render size of each quality mode is worked out the way OptiScaler
does it ((unsigned)(display / ratio) with 3.0, 2.0, 1.7, 1.5, 1.3 and 1.0 for ultra performance, performance, balanced,
quality, ultra quality and DLAA), and each of those rounded up to a multiple of 8, which some games render instead; m4r
picks the plan matching the size each frame is rendered at. Give --input for sizes a game picks on its own (dynamic resolution, custom scaling).
Plans contain Apple's translated kernels and weights made from your copy: keep them to yourself.

usage: install.py --resources <MetalFX Resources dir> --binary <MetalFX binary> (--game <dir with OptiScaler's dxgi.dll> | --plans <dir>)
                  [--display 1920x1080 ...] [--modes quality,balanced,performance,ultra,dlaa,ultra-quality]
                  [--input WxH ...] [--no-align8] [--depth reversed|forward|both] [--no-fg] [--no-denoiser] [--m4r-dll path] [--jobs N] [--force]
SPDX-License-Identifier: GPL-3.0-only
"""
import argparse, concurrent.futures, os, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RATIOS = {'ultra': 3.0, 'performance': 2.0, 'balanced': 1.7, 'quality': 1.5, 'ultra-quality': 1.3, 'dlaa': 1.0}


def size(text):
    w, h = text.lower().split('x')
    return int(w), int(h)


def render_size(display, ratio):
    return int(float(display[0]) / ratio), int(float(display[1]) / ratio)   # as OptiScaler: float / double, truncated


def need(tool):
    if shutil.which(tool) is None:
        sys.exit(f'{tool} not found (needed to translate the kernels: LLVM with the AMDGPU target, lld, glslang)')


def detect_gpu():
    """LLVM target of this machine's AMD GPU: on Windows from AMD's HIP runtime (part of the Adrenalin driver), otherwise from
    amdgpu's KFD topology (gfx_target_version 110001 -> gfx1101); the first GPU if there are several (pass --gpu to choose)."""
    if os.name == 'nt':
        import ctypes
        for dll in ('amdhip64_7.dll', 'amdhip64_6.dll'):
            try:
                hip = ctypes.WinDLL(dll)
            except OSError:
                continue
            n = ctypes.c_int(0)
            if hip.hipInit(0) or hip.hipGetDeviceCount(ctypes.byref(n)) or n.value < 1:
                return []
            found = []
            for i in range(n.value):
                props = ctypes.create_string_buffer(1472 + 1024)   # hipDeviceProp_tR0600; gcnArchName at 1160 (as runtime/hipvk)
                if not hip.hipGetDevicePropertiesR0600(props, i):
                    found.append(props.raw[1160:1416].split(b'\0')[0].decode().split(':')[0])
            return found
        return []
    import glob
    found = []
    for p in sorted(glob.glob('/sys/class/kfd/kfd/topology/nodes/*/properties')):
        try:
            for line in open(p):
                k, _, v = line.partition(' ')
                if k == 'gfx_target_version' and int(v) > 0:
                    v = int(v)
                    found.append(f'gfx{v // 10000}{(v // 100) % 100:x}{v % 100:x}')
        except OSError:
            pass
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--resources', required=True, help="MetalFX.framework/Versions/A/Resources of your macOS install")
    ap.add_argument('--binary', required=True, help='the MetalFX binary extracted from the dyld shared cache')
    ap.add_argument('--game', help="folder with OptiScaler's dxgi.dll; m4r/ is created in it")
    ap.add_argument('--plans', help='write the plans here instead of <game>/m4r/plans (and copy no m4r.dll)')
    ap.add_argument('--display', action='append', type=size, help='display resolution, e.g. 2560x1440 (default 1920x1080, 2560x1440, 3840x2160)')
    ap.add_argument('--modes', default='quality,balanced,performance,ultra', help='quality modes: ' + ', '.join(RATIOS))
    ap.add_argument('--input', action='append', type=size, help='extra render resolution (used with every --display)')
    ap.add_argument('--no-align8', dest='align8', action='store_false', help='no plans for mode sizes rounded up to a multiple of 8')
    ap.add_argument('--depth', choices=('reversed', 'forward', 'both'), default='both', help='depth conventions of the upscaler plans')
    ap.add_argument('--no-fg', action='store_true', help='no frame generation plans')
    ap.add_argument('--no-denoiser', action='store_true', help='no denoiser plans (Ray Reconstruction games)')
    ap.add_argument('--line-locking', action='store_true', help="upscaler plans with MetalFX's line locking (about 0.06 ms)")
    ap.add_argument('--m4r-dll', default=os.path.join(ROOT, 'build-win', 'm4r.dll'))
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument('--gpu', help='GPU the kernels are built for: gfx1030-gfx1036 (RDNA2), gfx1100-gfx1103, gfx1150-gfx1153 '
                                  '(RDNA3), gfx1200-gfx1201 (RDNA4); default: detected from this machine')
    ap.add_argument('--force', action='store_true', help='regenerate plans that exist')
    a = ap.parse_args()

    if not os.path.exists(os.path.join(a.resources, 'default.metallib')):
        sys.exit(f'{a.resources} has no default.metallib')
    if not os.path.isfile(a.binary):
        sys.exit(f'{a.binary} not found')
    if not a.game and not a.plans:
        sys.exit('pass --game or --plans')
    if not a.plans and not os.path.isfile(a.m4r_dll):
        sys.exit(f'{a.m4r_dll} not found (build it with: cmake --build build-win, or pass --m4r-dll)')
    native = os.name == 'nt'   # native Windows: kernels for HIP (code objects; air/m4r_air.py builds with clang when there is no llc)
    for t in ('llvm-as', 'llvm-dis', 'opt', 'clang' if native else 'llc', 'ld.lld', 'llvm-nm', 'llvm-objdump'):
        need(t)
    if not native and not os.path.isfile(os.path.join(ROOT, 'shaders', 'sacneh.spv')):   # a release ships it compiled
        need('glslangValidator')
    llc = subprocess.run(['clang' if native else 'llc', '--version'], capture_output=True, text=True).stdout
    major = next((int(w.split('.')[0]) for l in llc.split('\n') if ('LLVM version' in l or 'clang version' in l)
                  for w in l.split() if w[:1].isdigit()), 0)
    if major < 23:
        sys.exit(f'LLVM {major or "?"} found; m4r needs LLVM 23 or newer (it reads Apple\'s bitcode)')

    if not a.gpu:
        gpus = detect_gpu()
        if not gpus:
            sys.exit('could not detect the GPU (no amdgpu KFD topology); pass --gpu, e.g. --gpu gfx1101')
        a.gpu = gpus[0]
        print(f'GPU: {a.gpu}' + (f' (first of {", ".join(gpus)}; pass --gpu to choose)' if len(set(gpus)) > 1 else ''))

    pairs = []
    for disp in a.display or [(1920, 1080), (2560, 1440), (3840, 2160)]:
        for mode in [m for m in a.modes.split(',') if m]:
            if mode not in RATIOS:
                sys.exit(f'unknown mode {mode}; known: {", ".join(RATIOS)}')
            r = render_size(disp, RATIOS[mode])
            pairs.append((disp, r))
            # some games render the mode's size rounded up to a multiple of 8 (Ghost of Tsushima: 1505x847 -> 1512x848)
            r8 = (min(disp[0], (r[0] + 7) & ~7), min(disp[1], (r[1] + 7) & ~7))
            if a.align8 and r8 != r:
                pairs.append((disp, r8))
        for inp in a.input or []:
            pairs.append((disp, inp))
    pairs = sorted(set(pairs))
    for disp, inp in pairs:
        # MetalFX scales by 1x to 3x; truncation makes ultra performance land a hair above 3x (1280 / 426), which is fine
        if inp[0] > disp[0] or inp[1] > disp[1] or inp[0] * 3.1 < disp[0] or inp[1] * 3.1 < disp[1]:
            sys.exit(f'{inp[0]}x{inp[1]} -> {disp[0]}x{disp[1]}: MetalFX scales by 1x to 3x')

    plans_dir = a.plans or os.path.join(a.game, 'm4r', 'plans')
    os.makedirs(plans_dir, exist_ok=True)
    if native:   # hipvk's format-converting blit, next to the plans (runtime/hipvk/hipvk.h)
        sys.path.insert(0, ROOT)
        from air import m4r_air
        m4r_air.compile_cl(os.path.join(ROOT, 'runtime', 'hipvk', 'hipvk_blit.cl'), os.path.join(plans_dir, 'hipvk_blit.hsaco'), mcpu=a.gpu)
    if not a.plans:
        shutil.copyfile(a.m4r_dll, os.path.join(a.game, 'm4r', 'm4r.dll'))

    jobs = []   # (name, command)
    for disp, inp in pairs:
        tag = f'{disp[0]}x{disp[1]}-{inp[0]}x{inp[1]}'
        common = ['--resources', a.resources, '--output', f'{disp[0]}x{disp[1]}', '--input', f'{inp[0]}x{inp[1]}', '--gpu', a.gpu]
        for rev, suffix in ((1, ''), (0, '-fwd')):
            if (rev and a.depth == 'forward') or (not rev and a.depth == 'reversed'):
                continue
            jobs.append((f'bbr-{tag}{suffix}', [sys.executable, '-I', os.path.join(ROOT, 'tools', 'prepare_bbr.py'), *common,
                                                '--binary', a.binary, '--reversed-depth', str(rev)]
                         + (['--line-locking'] if a.line_locking else [])))
        if not a.no_denoiser and (disp, inp) == next(x for x in pairs if x[1] == inp):   # the denoiser only depends on the render size
            jobs.append((f'dn-{inp[0]}x{inp[1]}', [sys.executable, '-I', os.path.join(ROOT, 'tools', 'prepare_denoiser.py'), '--resources',
                                                  a.resources, '--input', f'{inp[0]}x{inp[1]}', '--stage', 'D', '--runtime', '--gpu', a.gpu]))
        if not a.no_denoiser:   # the V3 path (denoiser + UBFNet + V3 filter) writes the final image itself: one plan per display and render size
            jobs.append((f'dn3-{tag}', [sys.executable, '-I', os.path.join(ROOT, 'tools', 'prepare_denoiser.py'), '--resources', a.resources,
                                        '--input', f'{inp[0]}x{inp[1]}', '--output', f'{disp[0]}x{disp[1]}', '--stage', 'F', '--runtime',
                                        '--gpu', a.gpu]))
        if not a.no_fg:
            jobs.append((f'fg-{tag}', [sys.executable, '-I', os.path.join(ROOT, 'tools', 'prepare_fg.py'), *common]))

    def run(job):
        name, cmd = job
        out = os.path.join(plans_dir, name)
        if os.path.exists(os.path.join(out, 'plan.txt')) and not a.force:
            return name, 'kept'
        # built in <name>.part and moved into place when complete: an interrupted or failed build never looks finished
        part = out + '.part'
        if os.path.isdir(part):
            shutil.rmtree(part)
        r = subprocess.run(cmd + ['--out', part, '--force'], capture_output=True, text=True)
        if r.returncode:
            shutil.rmtree(part, ignore_errors=True)
            return name, 'FAILED\n' + (r.stderr or r.stdout)[-2000:]
        if os.path.isdir(out):
            shutil.rmtree(out)
        os.replace(part, out)
        return name, 'done'

    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        for name, state in pool.map(run, jobs):
            print(f'{name}: {state}', flush=True)
            failed += state.startswith('FAILED')
    print(f'{len(jobs) - failed} plans ready in {plans_dir}' + (f', {failed} failed' if failed else ''))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
