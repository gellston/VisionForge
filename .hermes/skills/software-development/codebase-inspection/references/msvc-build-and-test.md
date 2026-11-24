# MSVC: find MSBuild, build every config, run tests

Verified on Windows 11 with Visual Studio 2026 (v18). Use when reviewing a repo
whose solution has no CI you can read, so "the build passes" must be proven
locally. Generic detection for other toolchains (cargo/pytest/npm) lives in the
`requesting-code-review` skill's test step.

## 1. Locate MSBuild (never hardcode the path)

The install path varies by VS version and edition; discover it:

```bash
"/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe" \
  -latest -products '*' -requires Microsoft.Component.MSBuild \
  -find 'MSBuild\**\Bin\MSBuild.exe'
```

Returns a native Windows path (`C:\Program Files\...`). Pass it quoted to
MSBuild; git-bash does not translate paths for native tools, so prefer a
relative solution path (`visionforge/visionforge.sln`) from the repo root over
constructing one.

## 2. Build every configuration

A review that only builds one config misses config-specific breakage (Debug
asserts, `/MDd` vs `/MD`, export-macro differences):

```bash
MSBUILD="<vswhere result>"
"$MSBUILD" <repo>/<solution>.sln -p:Configuration=Release -p:Platform=x64 \
  -m -v:minimal -nologo 2>&1 | grep -iE "warning|error|-> "
```

Repeat for `Configuration=Debug` and, when the solution declares it,
`Platform=x86` (sln `x86` maps to project config `Win32`). At minimum run one
Debug + one Release — they use different CRTs, so one passing does not imply
the other.

**Always grep the full log for `warning|error`, never eyeball a truncated
window** — exit 0 means "no errors", not "no warnings". The `-> ` pattern
catches the `project.vcxproj -> path` lines that confirm each target actually
produced an artifact.

## 3. Output locations

- `Platform=x64` → `<solution-dir>/x64/<Configuration>/`
- `Platform=x86` (Win32) → `<solution-dir>/<Configuration>/` (no `x64` segment)

Both DLLs and test EXEs land in the same directory.

## 4. Run the test binaries

These projects use plain `main()` returning nonzero on failure — no test
framework to invoke:

```bash
cd <output-dir> && ./image_tests.exe; echo "exit=$?"
```

- **Run from the output directory**: the Windows loader resolves imported
  DLLs from the executable's own folder first; running via an absolute path
  from elsewhere fails with a missing-DLL error even though the build succeeded.
- Capture stdout too — these tests print match positions/scores, which is the
  evidence for "accuracy within tolerance" claims in the review.
- Report per-config exit codes in a table; "all three configs green" is the
  finding, not "it built".

## 5. Verify doc claims against build settings

When the README states runtime-library or flag expectations (`/MD`, `/MDd`,
C++ standard, toolset), grep the `.vcxproj` rather than trusting the prose:

```bash
grep -E "RuntimeLibrary|LanguageStandard|PlatformToolset|PreprocessorDefinitions" <repo>/**/*.vcxproj
```

An absent `<RuntimeLibrary>` tag means the MSBuild default (`/MD` Release,
`/MDd` Debug) — correct, but worth stating explicitly in the review since the
docs may claim the setting is pinned.
