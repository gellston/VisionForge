#include "shape_matching.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
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

int main()
{
    static_assert(std::is_nothrow_move_constructible_v<visionforge::v1::shape_matcher>);
    static_assert(std::is_standard_layout_v<visionforge::v1::shape_match_result>);
    try { test_matching(); test_validation(); }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
