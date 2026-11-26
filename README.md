# VisionForge

C++17 이미지 및 shape matching 라이브러리입니다. 공개 API는 `visionforge::v1` 네임스페이스에 있습니다.

## DLL 연결

1. Visual Studio 2026에서 [`visionforge/visionforge.sln`](visionforge/visionforge.sln)을 열고 사용할 구성의 `image` 프로젝트를 빌드합니다.
2. 사용하는 프로젝트의 추가 포함 디렉터리에 `visionforge/image/` 폴더를 지정합니다. 배포할 때는 `image.hpp`와 `export.hpp`를 함께 제공합니다.
3. 생성된 `image.lib`를 연결하고, 같은 구성의 `image.dll`을 실행 파일 옆에 배치합니다. x64 빌드 결과는 `visionforge/x64/Debug/` 또는 `visionforge/x64/Release/`에 생성됩니다.
4. DLL과 사용하는 프로그램의 아키텍처, Debug·Release 구성, 런타임 라이브러리를 맞춥니다. Release는 `/MD`, Debug는 `/MDd`를 사용합니다.

같은 솔루션에서는 `image_tests`처럼 `image` 프로젝트를 참조할 수 있습니다.
`visionforge_building_library`는 DLL을 빌드할 때만 사용하는 매크로이므로 사용하는 프로그램에는 정의하지 않습니다.

## image

헤더: [`visionforge/image/image.hpp`](visionforge/image/image.hpp)

| API                             | 설명                                |
| ------------------------------- | --------------------------------- |
| `image()`                       | 빈 이미지 생성                          |
| `image(width, height, channel)` | 버퍼를 생성하고 모든 바이트를 0으로 초기화          |
| `width()`                       | 가로 픽셀 수                           |
| `height()`                      | 세로 픽셀 수                           |
| `channel()`                     | 픽셀당 채널 수                          |
| `stride()`                      | 한 행의 바이트 수: `width() * channel()` |
| `data()`                        | 수정 가능한 픽셀 버퍼 시작 주소                |
| `data() const`                  | 읽기 전용 픽셀 버퍼 시작 주소                 |

채널당 1바이트이며, 각 픽셀의 채널은 연속으로 배치합니다. 행 패딩은 없고 RGB·BGR 등의 채널 의미는 지정하지 않습니다.
좌표 `(x, y)`의 채널 `c`는 `data()[y * stride() + x * channel() + c]`로 접근합니다. 좌표와 채널의 범위 검사는 호출자가 수행합니다.

복사하면 버퍼도 독립적으로 복사합니다. 이동된 원본과 기본 생성된 이미지는 크기·채널·stride가 0이고 `data()`는 `nullptr`입니다.
`data()`의 메모리는 직접 해제하지 않습니다. 객체가 파괴되거나 대입으로 버퍼가 교체되면 기존 포인터는 무효가 됩니다. 이동한 버퍼는 이동 대상 객체가 관리합니다.

크기와 채널은 모두 0보다 커야 합니다. 0을 전달하면 `std::invalid_argument`, 크기 곱셈이 `std::size_t` 범위를 넘으면 `std::length_error`가 발생합니다. 버퍼 크기나 메모리 부족에 따른 표준 컨테이너의 할당 관련 예외도 발생할 수 있습니다.

## 이미지 연산

`resize()`와 `crop()`은 `const` 멤버 함수이며 원본을 유지하고 독립된 픽셀 버퍼의 `image`를 반환합니다.
반환값을 변수에 저장하거나 대입해 사용합니다. 모든 채널 수를 지원하며 결과의 채널 수는 원본과 같습니다.

```cpp
auto resized = source.resize(640, 480);  // 기본 bilinear 보간
auto nearest = source.resize(320, 240,
    visionforge::v1::resize_interpolation::nearest);
auto roi = source.crop(10, 20, 100, 80);  // x, y, width, height
```

| 멤버 함수 | 설명 |
| --- | --- |
| `resize(width, height, interpolation)` | 지정한 픽셀 크기로 확대 또는 축소 |
| `crop(x, y, width, height)` | 좌상단 `(x, y)`에서 지정한 크기의 영역을 복사 |

`resize_interpolation::bilinear`가 기본값이며, 픽셀 중심을 기준으로 채널별 선형 보간하고 영상 경계에서는 가장자리 픽셀을 사용합니다.
`resize_interpolation::nearest`는 최근접 픽셀을 선택하므로 마스크나 레이블 영상의 값을 유지할 때 사용할 수 있습니다.
색상과 알파를 포함해 모든 채널을 독립적으로 처리합니다.

크기 인자는 모두 0보다 커야 합니다. `crop()`의 영역은 영상 안에 완전히 들어와야 하며, 바깥으로 넘어가는 부분을 자동으로 잘라내지 않습니다.
빈 이미지나 이동된 원본에서 호출하면 `std::logic_error`, 0 크기나 잘못된 보간 값에는 `std::invalid_argument`, 범위를 벗어난 crop 영역에는 `std::out_of_range`가 발생합니다.
결과 버퍼 크기 초과나 메모리 부족에 따른 생성·할당 예외도 발생할 수 있습니다.

## 스마트 포인터

| 타입                          | 정의                             |
| --------------------------- | ------------------------------ |
| `image::shared_image`       | `std::shared_ptr<image>`       |
| `image::shared_const_image` | `std::shared_ptr<const image>` |
| `image::unique_image`       | `std::unique_ptr<image>`       |

`shared_const_image`는 해당 포인터를 통한 수정을 제한합니다. 같은 객체를 가리키는 `shared_image`에서는 수정할 수 있습니다.

## 사용 예제

```cpp
#include "image.hpp"
#include <memory>

int main()
{
    visionforge::v1::image::shared_image shared =
        std::make_shared<visionforge::v1::image>(640, 480, 3);
    visionforge::v1::image::shared_const_image readonly = shared;
    visionforge::v1::image::unique_image unique =
        std::make_unique<visionforge::v1::image>(640, 480, 1);

    // (10, 20) 픽셀의 첫 번째 채널
    shared->data()[20 * shared->stride() + 10 * shared->channel()] = 255;

    // 독립된 버퍼로 복사
    visionforge::v1::image copied = *readonly;
    copied.data()[0] = 128;
    unique->data()[0] = 64;
}
```

## shape_matcher

헤더: [`visionforge/shape_matching/shape_matching.hpp`](visionforge/shape_matching/shape_matching.hpp)

템플릿과 원본 이미지에서 gradient 방향이 일치하는 위치와 회전을 찾아 가장 높은 점수의 결과 한 개를 반환합니다.
솔루션의 `shape_matching` 프로젝트를 빌드하면 `shape_matching.dll`과 `shape_matching.lib`가 생성됩니다.
`image`와 같은 `visionforge::v1` 네임스페이스를 사용합니다.

사용하는 프로젝트의 추가 포함 디렉터리에 `visionforge/image/`와 `visionforge/shape_matching/`를 지정하고, `image.lib`와 `shape_matching.lib`를 연결합니다.
실행 파일 옆에 같은 구성의 `image.dll`, `shape_matching.dll`을 배치합니다.
헤더를 배포할 때는 `image.hpp`, `export.hpp`, `shape_matching.hpp`, `shape_matching_export.hpp`를 함께 제공합니다.
`visionforge_building_shape_matching`은 DLL 빌드 전용이므로 사용하는 프로그램에는 정의하지 않습니다.

```cpp
#include "shape_matching.hpp"

visionforge::v1::shape_match_result find_shape(
    const visionforge::v1::image& template_image,
    const visionforge::v1::image& source)
{
    visionforge::v1::shape_match_options options;
    options.pyramid_levels = 3;             // 원본 포함: 원본, 1/2, 1/4
    options.rotation_min_degrees = -45.0;
    options.rotation_max_degrees = 45.0;
    options.rotation_step_degrees = 5.0;
    options.minimum_score = 0.7;

    visionforge::v1::shape_matcher matcher(options);
    matcher.set_template(template_image);   // 픽셀 데이터가 채워진 템플릿
    return matcher.match(source);
}
```

같은 템플릿으로 여러 영상을 검사하려면 `shape_matcher`를 유지하고 `match()`를 반복 호출합니다.
`set_template()`은 특징을 내부에 저장하므로 입력 템플릿의 수명에 의존하지 않습니다.

### 옵션

| `shape_match_options`   | 기본값      | 설명                                         |
| ----------------------- | -------- | ------------------------------------------ |
| `pyramid_levels`        | `3`      | 원본을 포함한 피라미드 레벨 수. 1~16           |
| `rotation_min_degrees`  | `-180.0` | 탐색 시작 각도                                   |
| `rotation_max_degrees`  | `180.0`  | 탐색 종료 각도                                   |
| `rotation_step_degrees` | `5.0`    | 회전 후보 간격. 0보다 커야 함                         |
| `minimum_score`         | `0.7`    | 일치 판정에 필요한 원본 해상도의 점수. 0 초과, 1 이하          |
| `gradient_threshold`    | `8.0`    | Sobel gradient를 8로 나눈 크기의 하한. 0 초과, 255 이하 |
| `maximum_features`      | `512`    | 레벨별 템플릿 특징 수 상한. 4~4096                    |
| `candidate_count`       | `32`     | 다음 레벨로 전달할 후보 수 상한. 1~256                  |
| `refine_position_radius`| `2`      | 각 후보를 재탐색할 반경(px). 0~256. 0이면 위치 보정 생략     |
| `refine_angle_steps`    | `1`      | 각 후보를 재탐색할 각도 스텝 수. 0~4094. 0이면 각도 보정 생략 |
| `edge_count_threshold`  | `0`      | 첫 스캔의 엣지 필터 막대. 0~1000000                          |
| `direct_refine`         | `false`  | 코스 스캔 결과를 최종 레벨로 한 번에 홉. 빠르지만 부정확              |

각도는 도 단위이며 화면 좌표에서 시계 방향이 양수입니다.
범위는 -360~360도 안에 있어야 하고, 시작값은 종료값 이하여야 하며, 전체 범위는 360도를 넘을 수 없습니다.
첫 번째(가장 저해상도) 스캔은 **적분이미지 사전 필터**를 거칩니다. 후보의 회전된 템플릿 바운딩 박스 안에 있는 엣지 픽셀 수를 네 번의 읽기로 세어,
그 수가 `edge_count_threshold`보다 적으면 스코어를 계산하지 않고 후보에서 제외합니다.
박스에 엣지가 하나도 없으면 모든 특징이 정확히 0점을 받아 스코어가 0이 되므로 제외해도 결과가 동일합니다(증명된 경로).
다만 이 필터는 **희소한 엣지 맵에서만 이득**입니다. 엣지가 조밀한 맵(원본 해상도의 노이즈 이미지 등)에서는 박스에
엣지가 항상 들어 있어 검사비만 들고 아무것도 걸러내지 못하므로, `edge_count_threshold = 0`인 기본값에서는
엣지 밀도를 재서 필터를 자동으로 끕니다(결과는 항상 동일합니다).
`edge_count_threshold`를 0보다 크게 지정하면 **밀도와 무관하게 항상 적용**되며, 이는
**속도를 위해 재현율을 거는 거래**라서 너무 높이면 실제 매치까지 걸러낼 수 있습니다.

`refine_position_radius`와 `refine_angle_steps`는 각 레벨에서 후보 한 개당 스코어를 `(2*반경+1)^2 * (2*스텝+1)`회 계산하게 하므로,
키우면 탐색 정밀도와 시간이 함께 늘어납니다. 0으로 줄이면 코스 레벨의 결과를 그대로 이어받아 가장 빠르지만,
코스 레벨이 정답에 못 미치면 보정할 기회가 없어 정확도가 떨어집니다.
시작값과 종료값을 같게 지정하면 해당 각도만 검색합니다.
간격이 범위를 정확히 나누지 못해도 종료 각도를 검색합니다. 한 바퀴의 양 끝처럼 같은 방향인 후보는 중복 제거합니다.
각도 후보 수는 최대 4096개이며 실수 옵션은 모두 유한한 값이어야 합니다.

`pyramid_levels`가 2 이상이면 가장 작은 레벨에서 위치·각도 후보를 찾고, 후보 좌표를 상위 해상도로 환산해 그 주변의 좁은 범위만 다시 탐색합니다.

`direct_refine = true`로 주면 이 과정에서 **중간 레벨을 건너뜁니다** — 코스 스캔의 후보를
`2^(pyramid_levels-1)`배 환산해 **가장 높은 해상도 레벨로 바로 이동**하고, 보정은 한 번만 합니다.
`pyramid_levels - 1`회였던 refine 단계가 1회로 줄어 **약 1.6배 빠릅니다**(3레벨 기준).

대신 코스 레벨의 위치 오차(최종 레벨 기준 `2^(pyramid_levels-1)`픽셀)가 중간에 좁혀지지 않으므로
**정확도가 떨어집니다**. 매치가 놓치기 시작하면 `refine_position_radius`를 키워서
오차 범위를 넓히세요.
이 과정을 원본 해상도까지 반복하며, 최종 위치와 점수는 원본에서 계산합니다. 후보 수를 제한하므로 전체 탐색과 결과가 다를 수 있습니다.
`pyramid_levels=1`이면 원본에서 전체 탐색합니다. 별도의 coarse-to-fine 옵션 없이 레벨 수로 탐색 방식을 결정합니다.
Gaussian 필터는 피라미드를 축소할 때 앨리어싱을 줄이는 용도로만 사용합니다.
회전은 지정한 각도 후보에서 선택하며, 크기 변화와 연속적인 세부 각도 보정은 지원하지 않습니다.

### 입력과 결과

입력은 최소 3x3 크기의 1·3·4채널 이미지입니다.
3·4채널은 앞의 세 채널 평균을 명도로 사용하고 네 번째 채널은 무시합니다.
템플릿은 사용하는 모든 피라미드 레벨에서 최소 5x5여야 하며, 레벨마다 임계값을 넘는 gradient 특징이 최소 4개 필요합니다.
작거나 특징이 약한 템플릿은 `pyramid_levels` 또는 `gradient_threshold`를 낮춰 사용합니다.
탐색 시 회전된 템플릿의 전체 사각형이 원본 영상 안에 들어와야 합니다.

| `shape_match_result` | 설명                                    |
| -------------------- | ------------------------------------- |
| `found`              | 최종 점수가 `minimum_score` 이상이면 `true`    |
| `x`, `y`             | 원본 영상에서 템플릿 중심 좌표. `found=true`일 때 사용 |
| `angle_degrees`      | 선택된 회전 각도. `found=true`일 때 사용         |
| `score`              | 0~1의 gradient 방향 일치 점수                |
| `top_left`, `top_right` | 회전된 템플릿의 좌상단·우상단 좌표 (`shape_match_point`) |
| `bottom_left`, `bottom_right` | 회전된 템플릿의 좌하단·우하단 좌표 (`shape_match_point`) |

네 꼭짓점의 `x`, `y`는 원본 영상의 실수 픽셀 좌표이며 `found=true`일 때 사용합니다.
템플릿의 `(0, 0)`부터 `(width - 1, height - 1)`까지의 픽셀 중심 사각형을 탐지 각도로 회전한 결과입니다.
꼭짓점 이름은 회전 전 템플릿 기준으로 유지합니다.
사각형을 그릴 때는 `top_left → top_right → bottom_right → bottom_left → top_left` 순서로 연결합니다.
결과 구조체의 크기가 변경되었으므로 DLL과 호출 프로그램을 함께 다시 빌드해야 합니다.

점수는 회전된 템플릿과 원본 gradient의 정규화 내적을 평균한 값입니다.
사라지거나 방향이 반대인 특징은 0점으로 계산합니다. 밝기 대비가 반전된 모양을 같은 모양으로 취급하지 않습니다.
위치는 픽셀 간격으로 검색하며, 짝수 크기 템플릿의 중심 좌표는 소수점을 포함할 수 있습니다.

`has_template()`으로 등록 여부, `options()`로 생성 시 지정한 옵션을 확인할 수 있습니다.
`clear_template()`은 등록한 템플릿을 제거합니다.
등록 전에 `match()`를 호출하면 `std::logic_error`, 잘못된 옵션·입력·템플릿에는 `std::invalid_argument`가 발생합니다.
새 템플릿 등록에 실패하면 기존 템플릿을 유지합니다.

### 회전된 템플릿 모서리

`rotated_template_corners()`는 등록한 템플릿을 각 후보 각도로 회전한 모서리 좌표를 반환합니다.
탐색에 사용하는 순서 그대로 각도 한 개당 항목 하나이며, 템플릿이 없으면 빈 목록입니다.

| `shape_rotation_corners` | 설명                                          |
| ------------------------ | ------------------------------------------- |
| `angle_degrees`          | 이 모서리가 대응하는 회전 각도                        |
| `top_left`, `top_right`  | 회전된 좌상단·우상단 좌표 (`shape_match_point`)         |
| `bottom_left`, `bottom_right` | 회전된 좌하단·우하단 좌표 (`shape_match_point`)    |

좌표는 **원본 영상이 아니라 템플릿 중심 기준**입니다. 어느 영상 위에 그릴지 정하려면 중심 좌표를 더합니다.

```cpp
const auto boxes = matcher.rotated_template_corners();   // 각도마다 1항목
for (const auto& box : boxes)
{
    // box.angle_degrees 로 각도 확인, 중심 (cx, cy)에 그릴 때:
    const double left   = cx + box.top_left.x;
    const double top    = cy + box.top_left.y;
    const double right  = cx + box.bottom_right.x;
    const double bottom = cy + box.bottom_right.y;
}
```

`shape_match_result`의 네 꼭짓점은 이미 원본 영상 좌표로 번역된 값이라 그대로 그릴 수 있고,
`rotated_template_corners()`의 항목에 중심을 더하면 같은 좌표가 됩니다.

### 캐시 동작

`set_template()`은 템플릿 피라미드의 모든 레벨에 대해 **회전된 특징·바운딩 박스·모서리를 미리 계산**해 보관합니다.
옵션은 생성 후 변경할 수 없으므로 각도 목록도 고정이라, 계산 결과를 `match()`가 끝까지 재사용합니다.
따라서 같은 템플릿으로 `match()`를 여러 번 호출할 때 회전 계산 비용은 발생하지 않습니다.
`clear_template()`은 이 캐시도 함께 비웁니다.

메모리는 레벨 수 × 각도 수 × 특징 수에 비례합니다. 기본 옵션(3레벨, 72각도, 특징 최대 512)에서
레벨당 최대 약 1.1 MiB이므로, `maximum_features`를 크게 올리면 캐시 크기도 그만큼 커집니다.

스마트 포인터 별칭은 `shape_matcher::shared_shape_matcher`, `shared_const_shape_matcher`, `unique_shape_matcher`입니다.
