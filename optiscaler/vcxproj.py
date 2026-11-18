#!/usr/bin/env python3
"""Print OptiScaler.vcxproj's Release|x64 build inputs as lines: SRC/INC/LIB/LIBDIR/DEF <value> (for CMake)."""
import os, re, sys
path = sys.argv[1]
proj = os.path.dirname(os.path.abspath(path))
sol = os.path.dirname(proj)
s = open(path, encoding='utf-8-sig').read()
fix = lambda p: p.replace('$(ProjectDir)', proj + '/').replace('$(SolutionDir)', sol + '/').replace('\\', '/').rstrip('/')
for src in re.findall(r'<ClCompile Include="([^"]+)"', s):
    print('SRC', src.replace('\\', '/'))
rel = s[s.index("Condition=\"'$(Configuration)|$(Platform)'=='Release|x64'\">\n    <IncludePath"):]
for p in re.search(r'<IncludePath>([^<]*)</IncludePath>', rel).group(1).split(';'):
    if p and '$(IncludePath)' not in p:
        print('INC', fix(p))
for p in re.search(r'<LibraryPath>([^<]*)</LibraryPath>', rel).group(1).split(';'):
    if p and '$(LibraryPath)' not in p:
        print('LIBDIR', fix(p))
item = s[s.index("<ItemDefinitionGroup Condition=\"'$(Configuration)|$(Platform)'=='Release|x64'\">"):]
item = item[:item.index('</ItemDefinitionGroup>')]
for lib in re.search(r'<AdditionalDependencies>([^<]*)</AdditionalDependencies>', item).group(1).split(';'):
    if lib and not lib.startswith('%'):
        print('LIB', lib)
for d in re.search(r'<PreprocessorDefinitions>([^<]*)</PreprocessorDefinitions>', item).group(1).split(';'):
    d = d.strip()
    if d and not d.startswith('%'):
        print('DEF', d)
