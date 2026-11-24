# VisionForge

C++17 이미지 처리 + shape matching 라이브러리. Visual Studio 솔루션 빌드, DLL 배포.

이 파일은 Hermes / Claude Code / Codex 등 이 저장소를 여는 모든 에이전트가 읽는 공통 지침이다.
프로젝트 규칙은 이 파일에, 절차형 지식은 `.hermes/skills/`에 둔다.

## 필수 규칙

- **빌드 검증 없이 "완료" 선언 금지.** 변경 후 MSBuild로 빌드하고 테스트 실행까지 끝낸다.
- 공개 API는 항상 `visionforge::v1` 네임스페이스. breaking change 시 v1 유지 여부를 먼저 논의.
- 헤더에 노출되는 멤버는 전부 `visionforge_api` / `visionforge_shape_api` 매크로로 내보낸다.
- pimpl 패턴(`struct impl; std::unique_ptr<impl>`) 유지. 헤더에 멤버 변수를 직접 노출하지 않는다.
- README.md의 API 표와 실제 구현을 항상 동기화한다. 표의 기본값·예외·제약은 코드에서 검증한 것만 적는다.
- 커밋 메시지: `feat:` / `fix:` / `docs:` / `test:` 접두어 (기존 로그 따름).

## 빌드 & 테스트

MSBuild 절대 경로 (PATH에 없음):

```
MSBUILD="/c/Program Files/Microsoft Visual Studio/18/Community/MSBuild/Current/Bin/MSBuild.exe"
```

```bash
# Release x64 (기본 검증 대상)
"$MSBUILD" visionforge/visionforge.sln -p:Configuration=Release -p:Platform=x64 -m -v:minimal -nologo

# 테스트 실행 (별도 프레임워크 없음, exit code 0 = 통과)
cd visionforge/x64/Release && ./image_tests.exe && ./shape_matching_tests.exe

# Debug x64 / Release x86 도 동일하게 검증해야 하는 변경:
#   -p:Configuration=Debug  -p:Platform=x64
#   -p:Configuration=Release -p:Platform=x86   (산출물은 visionforge/Release/)
```

- 경고 0개가 기준. 경고가 나오면 수정하거나 원인을 설명한다.
- 테스트는 `require()` 기반 자체 구현이다. `main()`이 0을 반환하면 통과.
- x64 산출물: `visionforge/x64/{Debug,Release}/`, x86: `visionforge/{Debug,Release}/` — 전부 gitignore됨.

## 아키텍처 요약

- `visionforge/image/` — `image` 클래스 (pimpl, `std::vector<uint8_t>` 버퍼, 채널당 1바이트 인터리브, 행 패딩 없음).
  `resize()` (bilinear 기본 / nearest), `crop()`. 강한 예외 안전성(임시 생성 후 swap).
- `visionforge/shape_matching/` — `shape_matcher`. 파이라미드 coarse-to-fine + 회전 각도 후보 탐색.
  파이프라인: `grayscale → downsample(5x5 binomial) → pyramid → Sobel(/8) → extract → rotations → full_search → refine`.
  점수 = 회전 특징과 원본 gradient의 클램프 내적 평균. 반전 극성은 0점.
- `*_tests/` — 이미지 연산/매칭 검증. 회전된 네 꼭짓점은 역변환으로 역검증.
- 빌드 매크로: `visionforge_building_library` / `visionforge_building_shape_matching`은 DLL 빌드 시에만 정의.
  사용 프로그램에는 정의하지 않는다.

## 주의할 엣지 케이스 (리뷰에서 발견)

- `crop()`는 원점 체크 후 감산해야 unsigned 오버플로가 없다. 순서를 바꾸지 말 것.
- `set_template()`은 로컬 vector에 다 만든 뒤 swap — 실패 시 기존 템플릿 보존. 이 불변식을 깨지 말 것.
- 피라미드 중심좌표는 `(W-1)/(2*scale)` 정의. 짜수 템플릿의 coarse→fine 2배 환산이 여기서 맞는다.
- `make_angles()`의 `span == 360.0`은 정확한 double 비교 — 근사값은 중복 제거 안 됨 (known, 정확성 무관).
- `shape_match_result` 크기 변경 시 DLL과 호출 프로그램을 함께 다시 빌드해야 한다 (README에 명시).

## 로컬 스킬 관리 (`.hermes/skills/`)

이 저장소의 스킬은 프로젝트 로컬 스킬로 격리 관리된다. 프로필(`~/.hermes/skills/`)과 섞이지 않고,
이 저장소에서 세션을 시작할 때만 로드되며, 커리터(autonomous skill maintenance)가 절대 건드리지 않는다.

- 위치: `.hermes/skills/<category>/<skill-name>/SKILL.md`
- 우선순위: 프로젝트 > 프로필 로컬 > `skills.create_dir` > `external_dirs` — 같은 이름이면 이 저장소 것이 이긴다.
- **처음 이 저장소에서 실행하면 trust가 필요하다:**
  ```bash
  hermes skills trust          # 저장소 루트에서 실행
  hermes skills trust <경로>   # 또는 경로 지정
  # 철회: hermes skills untrust
  ```
  신뢰 목록은 `skills.trusted_project_dirs` (`~/.hermes/config.yaml`)에 저장된다.
- 스킬 추가/수정은 이 저장소 안에서 `skill_manage`로 한다 (프로필 방향이 아니라). 커밋해서 다른 기기에 그대로 가져간다.
- 주의: 프로젝트 스킬은 내용이 바뀔 때마다 보안 스캔을 다시 받는다. 스캔에 걸리면 인덱스에서 quarantine된다.
- 프로젝트 스킬 디렉터리는 repo-owned: 에이전트가 만든 새 스킬도 기본적으로 `~/.hermes/skills/`로 가므로,
  여기에 남기려면 명시적으로 이 경로에 생성한다.

## 검증 체크리스트

1. `git status` — 의도한 파일만 변경되었는지 (빌드 산출물 유입 금지)
2. Release x64 빌드 경고 0
3. `image_tests` / `shape_matching_tests` 모두 exit 0
4. README API 표와 코드 불일치 없는지 확인
5. 커밋 후 `git push` (origin: `gellston/VisionForge`, 브랜치 `develop/1.0`)
