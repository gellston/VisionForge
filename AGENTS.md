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
- **MSBuild를 겹쳐 띄우지 말 것.** 같은 솔루션에 빌드가 2개 돌면
  `LNK1168: 쓰기용으로 shape_matching.dll을(를) 열 수 없습니다(오류 코드: 32)`가 난다.
  배경 검증을 `kill`해도 MSBuild 자식 프로세스가 남아 파일을 계속 잡는다 —
  `tasklist | grep MSBuild`로 남은 걸 확인하고 `MSYS2_ARG_CONV_EXCL='*' taskkill /PID <pid> /F`로 정리한 뒤
  다시 빌드한다. (`//PID`는 안 먹힌다.)
- **테스트가 멈추면 `user` CPU 시간을 볼 것.** 0에 가까우면 계산이 아니라 **대기**다 —
  모달 팝업이나 파일 잠금. `timeout 60 ./shape_matching_tests.exe`로 걸러서 돌린다.

## 아키텍처 요약

- `visionforge/image/` — `image` 클래스 (pimpl, `std::vector<uint8_t>` 버퍼, 채널당 1바이트 인터리브, 행 패딩 없음).
  `resize()` (bilinear 기본 / nearest), `crop()`. 강한 예외 안전성(임시 생성 후 swap).
- `visionforge/shape_matching/` — `shape_matcher`. 파이라미드 coarse-to-fine + 회전 각도 후보 탐색.
  파이프라인: `grayscale → downsample(5x5 binomial) → pyramid → Sobel(/8) → extract → rotations → full_search → refine`.
  점수 = 회전 특징과 원본 gradient의 클램프 내적 평균. 반전 극성은 0점.
  - **`downsample()`은 가분 필터로 계산한다.** 이항 커널 `w[dx]·w[dy]`가 outer product라 dx합이 dy와
    무관하므로, 행마다 한 번 계산한 뒤 그 행과 겹치는 출력 5행이 재사용한다(5슬롯 링버퍼 — 어떤 원본 행도
    출력 약 3개 동안만 살아있어 evict되지 않는다). 단일 채널에선 두 패스의 부분합이 전부 정수라
    **단일 패스 결과와 비트 동일**하다. 상세·성능 수치는 엣지 케이스 항목 참조.
  - **회전 캐시**: `set_template()`이 `rotated_levels[level][angle]`을 한 번에 구축한다.
    `match()`는 원본 쪽(pyramid·gradient)만 다시 계산한다.
  - **그라디언트 맵은 코스 레벨만 전체 구축**한다. `full_search()`가 그 맵 전체를 훑으므로 필요하지만,
    refine이 쓰는 fine 레벨은 `visit_regions()`가 후보 상자를 뽑고 `lazy_gradients()`가 **그 상자 안만**
    Sobel한다 — 나머지는 0으로 남고 어디서도 읽히지 않는다. 그런 맵에선 두 fast path가 꺼진다.
  - **레벨은 후보 상자 기반으로 만든다** (`match()` 경로). `pyramid_coarse()`는 레벨0 전체 평면을 아예
    만들지 않고 grayscale을 첫 다운샘플의 행 버퍼에 융합하며, refine 레벨은 `level_boxes()`가
    **원본에서 후보 상자만 크롭**해 `k`회 다운샘플한다. 상자에 **필터 반경만큼 마진**을 줘야 상자 픽셀이
    크롭 경계에 클램프되지 않고 체인 결과와 **비트 동일**이 유지된다. 크롭은 `x0`/`y0`(원점)과
    `level_width`/`level_height`(레벨 전체 크기)를 함께 들고 — 크롭 `width`로 레벨을 클램프하면 유니온이
    좁아져 채움 루프가 한 픽셀 밖으로 쓴다(힙 손상 `0xC0000374`).
  - **`direct_refine = true`면 중간 레벨을 건너뛴다.** 코스 후보를 `2^(pyramid_levels-1)`배 환산해
    최종 레벨로 한 번에 홉하고 보정은 한 번만(`target=0` 후 `break`). `refine()`/`visit_regions()`의
    `×2`가 `scale` 파라미터로 일반화돼 있다. 기본 `false`는 기존 점진적 동작 그대로.
  - `configure_edge_paths()`가 `edge_density`로 적분이미지 사전 필터·엣지 마스크를 켜고 끔.
  - `full_search()`는 y-불변항을 행 루프로 호이스트하고, 각도별로 `floor(x + K) = x + floor(K)`가
    구간 양 끝에서 확인될 때 나머지 `std::floor`도 루프 밖으로 뺀다. 불일치하면 원래 식으로 폴백하므로
    어느쪽이든 출력이 비트 동일하다.
- `*_tests/` — 이미지 연산/매칭 검증. 회전된 네 꼭짓점은 역변환으로 역검증.
  `shape_matching_tests`: matching / result_corners / rotation_corners / full_rotation_range /
  refine_options / edge_count_threshold / **border_object**(네 모서리 경계) /
  **direct_refine**(점진적 vs 한 번에 홉, 시간까지 출력) / validation.
- 빌드 매크로: `visionforge_building_library` / `visionforge_building_shape_matching`은 DLL 빌드 시에만 정의.
  사용 프로그램에는 정의하지 않는다.

## 주의할 엣지 케이스 (리뷰에서 발견)

- `crop()`는 원점 체크 후 감산해야 unsigned 오버플로가 없다. 순서를 바꾸지 말 것.
- `set_template()`은 로컬 vector에 다 만든 뒤 swap — 실패 시 기존 템플릿 보존. 이 불변식을 깨지 말 것.
- 피라미드 중심좌표는 `(W-1)/(2*scale)` 정의. 짜수 템플릿의 coarse→fine 2배 환산이 여기서 맞는다.
- **엣지 마스크/적분이미지의 빠른 경로는 `edge_density`에 따라 자동으로 켜진다** (`configure_edge_paths`).
  희소 맵(코스 레벨, d≈0.24%)에선 이득이지만, 조밀 맵(원본 해상도, d≈30%)에선 검사비만 들고 0건도 못 걸러서
  20% 역효과가 났던 적이 있다. 밀도 판단 없이 이 두 경로를 항상 켜지 말 것.
  단 `edge_count_threshold > 0`은 사용자가 명시한 요청이라 항상켜야 한다 (안 그러면 옵션이 무시된다).
- **Sobel 윈도우 경계는 항상 `width - 2` / `height - 2`다.** `gradients()`가 `x < width - 1`을 도는 건
  `value(1, 0)`이 `x+1`까지 읽기 때문이다. `lazy_gradients()`를 만들 때 `min(width - 1, box.x1)`로 잡았다가
  `input.pixels[width]`를 1개 넘겨 읽었다. Debug는 `_ITERATOR_DEBUG_LEVEL=2`가 막고 **모달 팝업으로
  프로세스가 클릭을 기다려 무한 대기**로 보이기 때문에 "테스트가 느리구나"로 오인하기 쉽고, Release는
  조용히 이웃 힙을 읽는다. 후보는 객체 위치에만 있는 게 아니라 `full_search`가 **맵 전체**에서 상위 32개를
  고르므로 가장자리 오탐 후보의 영역이 경계에 닿는다 — 중앙 배치도 안전하지 않다.
- 테스트 하네스 `main()`은 `_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE)`으로 그 팝업을 stderr로
  우회하고 `std::cout << std::unitbuf`를 한다. 재발 시 4초 만에 "vector subscript out of range"가 뜬다.
  `test_border_object()`는 네 모서리 배치로 경계 경로를 결정적으로 덮는다.
- **피라미드 `downsample()`의 5×5 이항 필터를 제거하지 말 것.** 실험 결과: 엣지 밀도가
  `0.01% → 30.87%`(700배)로 뛰어 `configure_edge_paths()`가 "이 맵은 필터가 이득 아니다"라고 정확히
  판단하고 사전 필터를 끈다 → `full_search`가 2,290만 위치 전부를 스코어링 → **26배 느려짐**
  (138ms → 3,654ms), 게다가 `border top-right` 테스트가 `found=0, angle=95`로 실패한다.
  필터는 얼리아싱 방지일 뿐 아니라 **사전 필터가 성립하는 전제조건**이다.
- **가분(separable)화 이득은 `N/3`** (N = tap 수). `N²` taps를 `src_y + out_y ≈ 3·out_y` 행으로 바꾸기 때문:
  `N=5 → 1.66×`, **`N=3 → 1.0× (무의미)** — 3×3은 가로 패스가 모든 원본 행을 돌아야 해서 절반이 상쇄된다.
  실제 `downsample()`은 5×5 가분 + 가중치 이중곱셈 제거 + float 누적으로 **pyramid 60 → 31.2ms (1.92×)**,
  5MiB 매치 135 → 94ms. 단일 채널에선 부분합이 전부 정수라(≤65,280 < 2²⁴) **출력이 비트 동일**하다.
  참고로 3×3으로 줄이면(9탭) 109ms로 가분 5×5보다 느리고 behavior까지 바뀐다.
- `make_angles()`의 `span == 360.0`은 정확한 double 비교 — 근사값은 중복 제거 안 됨 (known, 정확성 무관).
- **크롭 원점과 레벨 전체 크기를 반드시 구분할 것.** `lazy_gradients()`의 유니온을 크롭 `width`로
  클램프하면 `union_x1`이 좁아지고, 채움 루프는 여전히 레벨 `width`로 클램프해 `x = union_x1 + 1`에서
  `result.pixels`를 한 개 넘겨쓴다 → Release는 `0xC0000374`(STATUS_HEAP_CORRUPTION), Debug는
  `vector subscript out of range`. CRT 메시지엔 라인이 없으므로 의심 지점에 좌표를 출력하는
  임시 가드를 넣어 특정한 뒤 반드시 제거할 것.
- **`pyramid_coarse()`는 `validate_source()`를 직접 호출해야 한다.** 예전엔 `grayscale()`을 거쳐
  그쪽의 채널/크기 검증이 걸렸는데, 코스 레벨만 만드는 구조로 바뀌면서 검증이 빠졌다 —
  `match(image{})`(빈 이미지)와 `match(image(32,32,2))`(2채널)가 예외를 안 던졌다.
  `validate_source()`를 추출해 두 진입점이 공유한다.
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
   **Debug 통과를 생략하지 말 것** — `_ITERATOR_DEBUG_LEVEL=2`가 메모리 경계 버그를 잡는 유일한 구성이다.
   Release는 조용히 이웃 힙을 읽고 결과가 우연히 맞아 넘어가는 일이 있다.
4. README API 표와 코드 불일치 없는지 확인
5. 커밋 후 `git push` (origin: `gellston/VisionForge`, 브랜치 `develop/1.0`)
