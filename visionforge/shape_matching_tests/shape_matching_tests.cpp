#include "shape_matching.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#ifdef _MSC_VER
#include <crtdbg.h>
#endif
#include <utility>

namespace
{
using namespace visionforge::v1;
constexpr double pi = 3.14159265358979323846;

void require(bool value, const char* message)
{
    if (!value)
        throw std::runtime_error(message);
}

template <typename exception, typename action>
void require_throws(action&& operation)
{
    try { operation(); }
    catch (const exception&) { return; }
    throw std::runtime_error("Expected exception was not thrown");
}

image pattern(std::size_t width = 25, std::size_t height = 23)
{
    image result(width, height, 1);
    for (std::size_t y = 3; y <= 18; ++y)
        for (std::size_t x = 4; x <= 7; ++x)
            result.data()[y * result.stride() + x] = 230;
    for (std::size_t y = 15; y <= 18; ++y)
        for (std::size_t x = 7; x <= 20; ++x)
            result.data()[y * result.stride() + x] = 230;
    for (std::size_t y = 5; y <= 8; ++y)
        for (std::size_t x = 16; x <= 19; ++x)
            result.data()[y * result.stride() + x] = 170;
    return result;
}

// Inverse-warp rendering is independent of the matcher's feature transform.
image scene(const image& model, double cx, double cy, double angle, std::size_t channels = 1)
{
    image result(96, 88, channels);
    const double c = std::cos(angle * pi / 180.0), s = std::sin(angle * pi / 180.0);
    const auto pixel = [&](int x, int y) {
        if (x < 0 || y < 0 || x >= static_cast<int>(model.width()) || y >= static_cast<int>(model.height()))
            return 0.0;
        return static_cast<double>(model.data()[static_cast<std::size_t>(y) * model.stride() + x]);
    };
    for (std::size_t y = 0; y < result.height(); ++y)
    {
        for (std::size_t x = 0; x < result.width(); ++x)
        {
            const double dx = static_cast<double>(x) - cx, dy = static_cast<double>(y) - cy;
            const double tx = c * dx + s * dy + (static_cast<double>(model.width()) - 1.0) / 2.0;
            const double ty = -s * dx + c * dy + (static_cast<double>(model.height()) - 1.0) / 2.0;
            const int ix = static_cast<int>(std::floor(tx)), iy = static_cast<int>(std::floor(ty));
            const double fx = tx - ix, fy = ty - iy;
            const double value = (1.0 - fy) * ((1.0 - fx) * pixel(ix, iy) + fx * pixel(ix + 1, iy))
                               + fy * ((1.0 - fx) * pixel(ix, iy + 1) + fx * pixel(ix + 1, iy + 1));
            const auto byte = static_cast<std::uint8_t>(std::clamp(std::lround(value), 0L, 255L));
            for (std::size_t channel = 0; channel < channels; ++channel)
                result.data()[y * result.stride() + x * channels + channel] = channel == 3 ? 37 : byte;
        }
    }
    return result;
}

void check_location(const shape_match_result& match, double x, double y, double angle)
{
    std::cout << "match: " << match.found << " center=" << match.x << ',' << match.y
              << " angle=" << match.angle_degrees << " score=" << match.score << '\n';
    require(match.found, "Expected a match");
    require(std::hypot(match.x - x, match.y - y) <= 1.5, "Incorrect match position");
    require(std::abs(match.angle_degrees - angle) <= 5.0, "Incorrect match rotation");
    require(match.score >= 0.65 && match.score <= 1.0, "Invalid gradient score");
}

void test_matching()
{
    auto model = pattern();
    shape_match_options options;
    options.rotation_min_degrees = -45.0;
    options.rotation_max_degrees = 45.0;
    options.minimum_score = 0.65;
    shape_matcher matcher(options);
    require(!matcher.has_template(), "Unexpected initial template");
    matcher.set_template(model);
    require(matcher.has_template(), "Template not retained");
    require(matcher.options().pyramid_levels == 3, "Options not retained");

    const auto rotated = scene(model, 54.0, 49.0, 30.0);
    check_location(matcher.match(rotated), 54.0, 49.0, 30.0);
    check_location(matcher.match(scene(model, 42.0, 39.0, -35.0)), 42.0, 39.0, -35.0);
    check_location(matcher.match(scene(model, 33.0, 31.0, 0.0)), 33.0, 31.0, 0.0);
    check_location(matcher.match(scene(model, 62.0, 52.0, 45.0, 3)), 62.0, 52.0, 45.0);
    check_location(matcher.match(scene(model, 42.0, 45.0, -45.0, 4)), 42.0, 45.0, -45.0);
    require(!matcher.match(image(96, 88, 1)).found, "Blank source must not match");
    require(!matcher.match(image(5, 5, 1)).found, "Oversized template must not match");

    options.pyramid_levels = 1;
    shape_matcher exhaustive(options);
    exhaustive.set_template(model);
    const auto full = exhaustive.match(rotated);
    const auto refined = matcher.match(rotated);
    check_location(full, 54.0, 49.0, 30.0);
    require(refined.x == full.x && refined.y == full.y && refined.angle_degrees == full.angle_degrees
            && std::abs(refined.score - full.score) < 1e-12,
            "Pyramid refinement must evaluate the final pose in the original image");

    options.pyramid_levels = 2;
    shape_matcher two_levels(options);
    two_levels.set_template(model);
    check_location(two_levels.match(rotated), 54.0, 49.0, 30.0);

    options.rotation_min_degrees = 12.0;
    options.rotation_max_degrees = 12.0;
    shape_matcher fixed_angle(options);
    fixed_angle.set_template(model);
    const auto fixed = fixed_angle.match(scene(model, 50.0, 45.0, 12.0));
    check_location(fixed, 50.0, 45.0, 12.0);
    require(fixed.angle_degrees == 12.0, "Rotation must respect fixed range");

    // Non-divisible ranges must include the maximum angle.
    options.rotation_min_degrees = 0.0;
    options.rotation_max_degrees = 17.0;
    options.rotation_step_degrees = 10.0;
    shape_matcher endpoint(options);
    endpoint.set_template(model);
    const auto end = endpoint.match(scene(model, 50.0, 45.0, 17.0));
    check_location(end, 50.0, 45.0, 17.0);
    require(end.angle_degrees == 17.0, "Maximum angle was not searched");

    // A linear brightness change preserves normalized gradient direction.
    options.pyramid_levels = 1;
    options.rotation_min_degrees = 0.0;
    options.rotation_max_degrees = 0.0;
    options.minimum_score = 0.95;
    shape_matcher brightness(options);
    brightness.set_template(model);
    auto illuminated = scene(model, 33.0, 31.0, 0.0);
    for (std::size_t i = 0; i < illuminated.stride() * illuminated.height(); ++i)
        illuminated.data()[i] = static_cast<std::uint8_t>(illuminated.data()[i] / 2 + 20);
    const auto bright = brightness.match(illuminated);
    check_location(bright, 33.0, 31.0, 0.0);
    require(std::abs(bright.score - 1.0) < 1e-12, "Scoring must use gradient direction, not intensity");
    for (std::size_t i = 0; i < illuminated.stride() * illuminated.height(); ++i)
        illuminated.data()[i] = static_cast<std::uint8_t>(255 - illuminated.data()[i]);
    require(!brightness.match(illuminated).found, "Reversed edge polarity must not receive a perfect score");
    options.minimum_score = 0.65;

    auto even_model = pattern(26, 24);
    options.rotation_min_degrees = 0.0;
    options.rotation_max_degrees = 0.0;
    options.pyramid_levels = 3;
    shape_matcher even(options);
    even.set_template(even_model);
    check_location(even.match(scene(even_model, 50.5, 45.5, 0.0)), 50.5, 45.5, 0.0);

    const auto unrotated_scene = scene(model, 54.0, 49.0, 0.0);
    const auto template_roi = unrotated_scene.crop(42, 38, 25, 23);
    shape_matcher roi_matcher(matcher.options());
    roi_matcher.set_template(template_roi);
    check_location(roi_matcher.match(rotated), 54.0, 49.0, 30.0);

    // Caller changes and failed retraining must not invalidate the cached template.
    auto copied = matcher;
    std::fill_n(model.data(), model.stride() * model.height(), std::uint8_t{0});
    require_throws<std::invalid_argument>([&] { matcher.set_template(model); });
    check_location(matcher.match(rotated), 54.0, 49.0, 30.0);
    matcher.clear_template();
    require(!matcher.has_template(), "Template was not cleared");
    check_location(copied.match(rotated), 54.0, 49.0, 30.0);
    auto moved = std::move(copied);
    require(!copied.has_template(), "Moved-from matcher must be empty");
    check_location(moved.match(rotated), 54.0, 49.0, 30.0);
    shape_matcher assigned;
    assigned = moved;
    require(assigned.has_template(), "Copy assignment lost template");
    shape_matcher move_assigned;
    move_assigned = std::move(assigned);
    require(move_assigned.has_template() && !assigned.has_template(), "Move assignment lost template");
}

void test_result_corners()
{
    // Inverse-transform reported corners into template coordinates to check both
    // their identities and their original-resolution extent after refinement.
    for (const std::size_t levels : {std::size_t{1}, std::size_t{3}})
    {
        for (const bool even : {false, true})
        {
            const auto model = even ? pattern(26, 24) : pattern();
            for (const double angle : {0.0, 30.0, -30.0, 90.0, -90.0, 180.0})
            {
                shape_match_options options;
                options.pyramid_levels = levels;
                options.rotation_min_degrees = angle;
                options.rotation_max_degrees = angle;
                options.minimum_score = 0.65;
                shape_matcher matcher(options);
                matcher.set_template(model);
                const double cx = even ? 50.5 : 50.0;
                const double cy = even ? 45.5 : 45.0;
                const auto result = matcher.match(scene(model, cx, cy, angle));
                check_location(result, cx, cy, angle);
                const double c = std::cos(result.angle_degrees * pi / 180.0);
                const double s = std::sin(result.angle_degrees * pi / 180.0);
                const auto check_corner = [&](const shape_match_point& point, double tx, double ty) {
                    const double dx = point.x - result.x, dy = point.y - result.y;
                    require(std::abs(c * dx + s * dy + (model.width() - 1.0) / 2.0 - tx) < 1e-9
                         && std::abs(-s * dx + c * dy + (model.height() - 1.0) / 2.0 - ty) < 1e-9,
                            "Incorrect rotated template corner");
                };
                check_corner(result.top_left, 0.0, 0.0);
                check_corner(result.top_right, model.width() - 1.0, 0.0);
                check_corner(result.bottom_left, 0.0, model.height() - 1.0);
                check_corner(result.bottom_right, model.width() - 1.0, model.height() - 1.0);

                const auto missing = matcher.match(image(96, 88, 1));
                require(!missing.found && missing.top_left.x == 0.0 && missing.top_left.y == 0.0
                     && missing.top_right.x == 0.0 && missing.top_right.y == 0.0
                     && missing.bottom_left.x == 0.0 && missing.bottom_left.y == 0.0
                     && missing.bottom_right.x == 0.0 && missing.bottom_right.y == 0.0,
                        "Empty match must return default corners");
            }
        }
    }
}

void test_validation()
{
    shape_matcher matcher;
    require_throws<std::logic_error>([&] { (void)matcher.match(image(32, 32, 1)); });
    require_throws<std::invalid_argument>([&] { matcher.set_template(image{}); });
    require_throws<std::invalid_argument>([&] { matcher.set_template(image(25, 23, 2)); });
    require_throws<std::invalid_argument>([&] { matcher.set_template(image(25, 23, 1)); });
    require_throws<std::invalid_argument>([&] { matcher.set_template(pattern(25, 23)); matcher.set_template(image(5, 5, 1)); });
    require_throws<std::invalid_argument>([&] { (void)matcher.match(image{}); });
    require_throws<std::invalid_argument>([&] { (void)matcher.match(image(32, 32, 2)); });
    const auto invalid = [](shape_match_options options) {
        require_throws<std::invalid_argument>([&] { shape_matcher bad(options); });
    };
    shape_match_options options;
    options.pyramid_levels = 0; invalid(options);
    options = {}; options.pyramid_levels = 17; invalid(options);
    options = {}; options.rotation_step_degrees = 0.0; invalid(options);
    options = {}; options.rotation_min_degrees = 45.0; options.rotation_max_degrees = -45.0; invalid(options);
    options = {}; options.rotation_step_degrees = 0.00001; invalid(options);
    options = {}; options.rotation_min_degrees = std::numeric_limits<double>::quiet_NaN(); invalid(options);
    options = {}; options.minimum_score = 1.1; invalid(options);
    options = {}; options.gradient_threshold = 0.0; invalid(options);
    options = {}; options.maximum_features = 0; invalid(options);
    options = {}; options.candidate_count = 0; invalid(options);
}
}

void test_full_rotation_range()
{
    // The rotation range is an option; exercise it explicitly at -180..180 with
    // every other option left at its default.
    const auto model = pattern();
    shape_match_options options;
    options.rotation_min_degrees = -180.0;
    options.rotation_max_degrees = 180.0;
    shape_matcher matcher(options);
    require(matcher.options().rotation_min_degrees == -180.0
         && matcher.options().rotation_max_degrees == 180.0,
            "Rotation range must be readable back from options()");
    matcher.set_template(model);

    // A full turn of 5 degree steps drops one endpoint: -180 and 180 describe the
    // same orientation, so the candidate list is -180..175.
    const auto corners = matcher.rotated_template_corners();
    require(corners.size() == 72, "Full 360 degree range must yield 72 candidates");
    require(corners.front().angle_degrees == -180.0, "Full range must start at -180");
    require(corners.back().angle_degrees == 175.0, "Full range must stop at 175");
    for (std::size_t i = 1; i < corners.size(); ++i)
        require(std::abs((corners[i].angle_degrees - corners[i - 1].angle_degrees) - 5.0) < 1e-12,
                "Full range candidates must stay 5 degrees apart");

    // Every other option is still at its default.
    require(matcher.options().pyramid_levels == 3
         && matcher.options().rotation_step_degrees == 5.0
         && matcher.options().minimum_score == 0.7
         && matcher.options().gradient_threshold == 8.0
         && matcher.options().maximum_features == 512
         && matcher.options().candidate_count == 32
         && matcher.options().refine_position_radius == 2
         && matcher.options().refine_angle_steps == 1,
            "Untouched options must keep their defaults");

    // Match across the whole circle. 180 is skipped: its grid slot is -180,
    // and the tolerance in check_location is an absolute degree difference.
    const double cx = 50.0, cy = 45.0;
    for (const double angle : {-180.0, -175.0, -135.0, -90.0, -45.0,
                               0.0, 30.0, 45.0, 90.0, 135.0, 175.0})
    {
        const auto result = matcher.match(scene(model, cx, cy, angle));
        check_location(result, cx, cy, angle);
        require(result.score >= 0.7, "Full-range match must clear the default minimum_score");
    }
}

void test_rotation_corners()
{
    // No template yet: no rotations to report.
    shape_matcher empty;
    require(empty.rotated_template_corners().empty(), "Empty matcher must report no rotation corners");

    const auto model = pattern();
    shape_match_options options;
    options.rotation_min_degrees = -30.0;
    options.rotation_max_degrees = 30.0;
    options.rotation_step_degrees = 15.0;
    options.minimum_score = 0.65;
    shape_matcher matcher(options);
    matcher.set_template(model);
    const auto corners = matcher.rotated_template_corners();
    // -30, -15, 0, 15, 30 -> five angles.
    require(corners.size() == 5, "Rotation corners must cover every configured angle");

    const double half_w = (model.width() - 1.0) / 2.0;
    const double half_h = (model.height() - 1.0) / 2.0;
    for (std::size_t i = 0; i < corners.size(); ++i)
    {
        require(std::abs(corners[i].angle_degrees - (-30.0 + 15.0 * static_cast<double>(i))) < 1e-12,
                "Rotation corner angles must follow the configured sequence");
        const double c = std::cos(corners[i].angle_degrees * pi / 180.0);
        const double s = std::sin(corners[i].angle_degrees * pi / 180.0);
        // Corners are template-center relative: rotate the unrotated offsets and compare.
        const auto check = [&](const shape_match_point& point, double tx, double ty) {
            require(std::abs(point.x - (c * tx - s * ty)) < 1e-9
                 && std::abs(point.y - (s * tx + c * ty)) < 1e-9,
                    "Rotation corner does not match the rotation matrix");
        };
        check(corners[i].top_left, -half_w, -half_h);
        check(corners[i].top_right, half_w, -half_h);
        check(corners[i].bottom_left, -half_w, half_h);
        check(corners[i].bottom_right, half_w, half_h);
        // A box's diagonals stay equal length under rotation.
        const double d1 = std::hypot(corners[i].top_right.x - corners[i].bottom_left.x,
                                     corners[i].top_right.y - corners[i].bottom_left.y);
        const double d2 = std::hypot(corners[i].bottom_right.x - corners[i].top_left.x,
                                     corners[i].bottom_right.y - corners[i].top_left.y);
        require(std::abs(d1 - d2) < 1e-9 && std::abs(d1 - std::hypot(2.0 * half_w, 2.0 * half_h)) < 1e-9,
                "Rotation must preserve the template diagonal");
    }

    // Reported corners must agree with the winning match's translated corners.
    const auto result = matcher.match(scene(model, 54.0, 49.0, 15.0));
    check_location(result, 54.0, 49.0, 15.0);
    std::size_t best = 0;
    for (std::size_t i = 0; i < corners.size(); ++i)
        if (std::abs(corners[i].angle_degrees - result.angle_degrees) < 1e-12)
            best = i;
    require(std::abs(corners[best].top_left.x + result.x - result.top_left.x) < 1e-9
         && std::abs(corners[best].top_left.y + result.y - result.top_left.y) < 1e-9,
            "Rotation corners must translate into the match result corners");

    // Clearing drops the cache along with the template.
    matcher.clear_template();
    require(matcher.rotated_template_corners().empty(), "Cleared matcher must report no rotation corners");
}

void test_refine_options()
{
    const auto model = pattern();
    const auto rotated = scene(model, 54.0, 49.0, 30.0);

    // Defaults keep the historical neighborhood: +/-2 px and +/-1 angle step.
    shape_matcher defaults;
    require(defaults.options().refine_position_radius == 2
         && defaults.options().refine_angle_steps == 1,
            "Refine defaults must stay at 2 px and 1 angle step");

    // Widening the neighborhood still finds the same pose.
    shape_match_options wide;
    wide.refine_position_radius = 4;
    wide.refine_angle_steps = 2;
    shape_matcher wide_matcher(wide);
    wide_matcher.set_template(model);
    check_location(wide_matcher.match(rotated), 54.0, 49.0, 30.0);

    // A zero neighborhood skips local search entirely: refine scores only the
    // doubled coarse position at the coarse angle. That is valid input and must
    // run, but it can only succeed when the coarse level already landed on the
    // answer, so the pose itself is not guaranteed here.
    shape_match_options pinned;
    pinned.refine_position_radius = 0;
    pinned.refine_angle_steps = 0;
    shape_matcher pinned_matcher(pinned);
    pinned_matcher.set_template(model);
    const auto pinned_result = pinned_matcher.match(rotated);
    require(std::isfinite(pinned_result.score) && pinned_result.score >= 0.0
         && pinned_result.score <= 1.0,
            "Zero-radius refine must produce a finite score in range");

    // Out-of-range neighborhoods are rejected up front.
    using setter = void (*)(shape_match_options&);
    const setter invalid[] = {
        [](shape_match_options& o) { o.refine_position_radius = 257; },
        [](shape_match_options& o) { o.refine_angle_steps = 4095; },
    };
    for (const auto bad : invalid)
    {
        shape_match_options options;
        bad(options);
        require_throws<std::invalid_argument>([&] { shape_matcher rejected(options); });
    }
}

void test_edge_count_threshold()
{
    const auto model = pattern();
    const auto source = scene(model, 54.0, 49.0, 30.0);

    // Default keeps the provable floor: drop only boxes holding no edge at all.
    shape_matcher defaults;
    require(defaults.options().edge_count_threshold == 0,
            "Edge count threshold must default to 0");
    defaults.set_template(model);
    check_location(defaults.match(source), 54.0, 49.0, 30.0);

    // 1 asks for the same floor the default already applies.
    shape_match_options one;
    one.edge_count_threshold = 1;
    shape_matcher one_matcher(one);
    one_matcher.set_template(model);
    check_location(one_matcher.match(source), 54.0, 49.0, 30.0);

    // A bar no bounding box can reach filters every candidate out of the scan.
    shape_match_options impossible;
    impossible.edge_count_threshold = 1000;
    shape_matcher none(impossible);
    none.set_template(model);
    require(!none.match(source).found,
            "An unreachable edge count threshold must reject every candidate");

    // Out-of-range thresholds are refused up front.
    shape_match_options too_high;
    too_high.edge_count_threshold = 1000001;
    require_throws<std::invalid_argument>([&] { shape_matcher rejected(too_high); });
}

// The object is placed hard against each corner so refine()'s candidate
// regions reach the map border. A Sobel window reads one pixel either side, so
// lazy_gradients() must clamp to width - 2 / height - 2 exactly as gradients()
// does. Clamping to width - 1 instead let value(1, 0) read input.pixels[width]
// -- one past the end, which Debug's checked operator[] rejects with a dialog
// while Release silently reads neighbouring heap. Regions never reached the
// border in the other tests, so only a placement like this exposes it.
void test_border_object()
{
    const auto model = pattern();
    shape_matcher matcher;
    matcher.set_template(model);
    const struct
    {
        double cx;
        double cy;
        const char* where;
    } placements[] = {
        {14.0, 13.0, "top-left"},
        {82.0, 13.0, "top-right"},
        {14.0, 75.0, "bottom-left"},
        {82.0, 75.0, "bottom-right"},
    };
    for (const auto& placement : placements)
    {
        std::cout << "border " << placement.where << ": ";
        const auto result = matcher.match(scene(model, placement.cx, placement.cy, 0.0));
        check_location(result, placement.cx, placement.cy, 0.0);
    }
}

int main()
{
#ifdef _MSC_VER
    // Debug's checked STL paths report through a modal dialog by default, so a
    // failure leaves the test sitting on a click instead of exiting. Send those
    // reports to stderr so the run fails loudly and cleanly on every platform
    // the suite is expected to pass on.
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
#endif
    // Unbuffered, so a hard abort inside a checked path still shows which test
    // was running instead of losing the buffered lines.
    std::cout << std::unitbuf;
    static_assert(std::is_nothrow_move_constructible_v<visionforge::v1::shape_matcher>);
    static_assert(std::is_standard_layout_v<visionforge::v1::shape_match_result>);
    try { test_matching(); test_result_corners(); test_rotation_corners();
          test_full_rotation_range(); test_refine_options(); test_edge_count_threshold();
          test_border_object();
          test_validation(); }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
