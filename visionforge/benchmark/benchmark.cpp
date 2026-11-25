#include "image.hpp"
#include "shape_matching.hpp"

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <exception>
#include <random>
#include <vector>

using namespace visionforge::v1;
using clock_type = std::chrono::steady_clock;

namespace
{
double milliseconds_since(clock_type::time_point start)
{
    return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
}

// 25 x 23 template matching the shape used by shape_matching_tests.
image make_template()
{
    image result(25, 23, 1);
    auto* data = result.data();
    const auto fill = [&](std::size_t x0, std::size_t x1, std::size_t y0, std::size_t y1, std::uint8_t value) {
        for (std::size_t y = y0; y <= y1; ++y)
            for (std::size_t x = x0; x <= x1; ++x)
                data[y * result.stride() + x] = value;
    };
    fill(4, 7, 3, 18, 230);
    fill(7, 20, 15, 18, 230);
    fill(16, 19, 5, 8, 170);
    return result;
}

// Same pattern scaled by an integer factor. pyramid_levels = 4 needs a template
// that still measures at least 5 x 5 three halvings down, i.e. 33 x 33 or more;
// the default 25 x 23 collapses to 4 x 3 and set_template() rejects it.
image make_scaled_template(std::size_t scale)
{
    if (scale <= 1)
        return make_template();
    image result(25 * scale, 23 * scale, 1);
    auto* data = result.data();
    const auto fill = [&](std::size_t x0, std::size_t x1, std::size_t y0, std::size_t y1, std::uint8_t value) {
        for (std::size_t y = y0; y <= y1; ++y)
            for (std::size_t x = x0; x <= x1; ++x)
                data[y * result.stride() + x] = value;
    };
    fill(4 * scale, 7 * scale, 3 * scale, 18 * scale, 230);
    fill(7 * scale, 20 * scale, 15 * scale, 18 * scale, 230);
    fill(16 * scale, 19 * scale, 5 * scale, 8 * scale, 170);
    return result;
}

// Roughly textured source with the template stamped at a known location.
image make_source(std::size_t width, std::size_t height, const image& stamp, std::size_t at_x, std::size_t at_y)
{
    image result(width, height, 1);
    std::mt19937 generator(12345);
    std::uniform_int_distribution<int> noise(0, 40);
    auto* data = result.data();
    for (std::size_t y = 0; y < height; ++y)
        for (std::size_t x = 0; x < width; ++x)
            data[y * result.stride() + x] = static_cast<std::uint8_t>(noise(generator));

    // Low-frequency gradient so the background has structure without strong edges.
    for (std::size_t y = 0; y < height; ++y)
        for (std::size_t x = 0; x < width; ++x)
            data[y * result.stride() + x] = static_cast<std::uint8_t>(
                data[y * result.stride() + x] + (x * 40) / width);

    for (std::size_t y = 0; y < stamp.height(); ++y)
        for (std::size_t x = 0; x < stamp.width(); ++x)
            data[(at_y + y) * result.stride() + (at_x + x)] = stamp.data()[y * stamp.stride() + x];
    return result;
}
}

int run_benchmark(int argc, char** argv)
{
    // Default: 2560 x 2048 x 1 = 5,242,880 bytes = exactly 5 MiB.
    std::size_t width = 2560;
    std::size_t height = 2048;
    std::size_t runs = 3;
    double rotation_step = 5.0;
    std::size_t pyramid_levels = 3;
    std::size_t refine_position_radius = 2;
    std::size_t refine_angle_steps = 1;
    double rotation_min = -180.0;
    double rotation_max = 180.0;
    std::size_t edge_count_threshold = 0;
    std::size_t template_scale = 1;
    if (argc >= 3)
    {
        width = static_cast<std::size_t>(std::atol(argv[1]));
        height = static_cast<std::size_t>(std::atol(argv[2]));
    }
    if (argc >= 4)
        runs = static_cast<std::size_t>(std::atol(argv[3]));
    if (argc >= 5)
        rotation_step = std::atof(argv[4]);
    if (argc >= 6)
        pyramid_levels = static_cast<std::size_t>(std::atol(argv[5]));
    if (argc >= 7)
        refine_position_radius = static_cast<std::size_t>(std::atol(argv[6]));
    if (argc >= 8)
        refine_angle_steps = static_cast<std::size_t>(std::atol(argv[7]));
    if (argc >= 9)
        rotation_min = std::atof(argv[8]);
    if (argc >= 10)
        rotation_max = std::atof(argv[9]);
    if (argc >= 11)
        edge_count_threshold = static_cast<std::size_t>(std::atol(argv[10]));
    if (argc >= 12)
        template_scale = static_cast<std::size_t>(std::atol(argv[11]));
    if (width == 0 || height == 0 || runs == 0 || pyramid_levels == 0)
    {
        std::puts("usage: benchmark [width height [runs [rotation_step [pyramid_levels "
                  "[refine_radius [refine_angle_steps [rot_min [rot_max "
                  "[edge_count_threshold]]]]]]]]]");
        return 2;
    }

    const auto template_image = make_scaled_template(template_scale);
    const auto source = make_source(width, height, template_image, width / 2, height / 2);
    std::printf("source: %zu x %zu x 1 = %.2f MiB, template=%zux%zu, runs=%zu, "
                "rot=[%.1f,%.1f] step=%.1f deg, levels=%zu, refine=+-%zupx/+-%zuang, "
                "edge_threshold=%zu\n",
                width, height, static_cast<double>(width * height) / (1024.0 * 1024.0),
                template_image.width(), template_image.height(),
                runs, rotation_min, rotation_max, rotation_step,
                pyramid_levels, refine_position_radius, refine_angle_steps,
                edge_count_threshold);

    shape_match_options options;
    options.rotation_min_degrees = rotation_min;
    options.rotation_max_degrees = rotation_max;
    options.rotation_step_degrees = rotation_step;
    options.pyramid_levels = pyramid_levels;
    options.refine_position_radius = refine_position_radius;
    options.refine_angle_steps = refine_angle_steps;
    options.edge_count_threshold = edge_count_threshold;
    shape_matcher matcher(options);

    double set_template_total = 0.0;
    for (std::size_t i = 0; i < runs; ++i)
    {
        const auto start = clock_type::now();
        matcher.set_template(template_image);
        set_template_total += milliseconds_since(start);
    }
    std::printf("set_template : %10.3f ms (avg over %zu runs)\n", set_template_total / runs, runs);

    shape_match_result result;
    double match_total = 0.0;
    double min_match = 1e18;
    double max_match = 0.0;
    for (std::size_t i = 0; i < runs; ++i)
    {
        const auto start = clock_type::now();
        result = matcher.match(source);
        const auto elapsed = milliseconds_since(start);
        match_total += elapsed;
        if (elapsed < min_match) min_match = elapsed;
        if (elapsed > max_match) max_match = elapsed;
    }
    std::printf("match        : %10.3f ms avg, min %.3f, max %.3f (%zu runs)\n",
                match_total / runs, min_match, max_match, runs);
    std::printf("result       : found=%d center=(%.1f, %.1f) angle=%.1f score=%.4f\n",
                result.found ? 1 : 0, result.x, result.y, result.angle_degrees, result.score);
    std::printf("corners      : tl=(%.1f,%.1f) tr=(%.1f,%.1f) bl=(%.1f,%.1f) br=(%.1f,%.1f)\n",
                result.top_left.x, result.top_left.y, result.top_right.x, result.top_right.y,
                result.bottom_left.x, result.bottom_left.y, result.bottom_right.x, result.bottom_right.y);
    return result.found ? 0 : 1;
}

// Report option errors (bad pyramid level for the template size, out-of-range
// values) instead of letting an uncaught exception abort without a message.
int main(int argc, char** argv)
{
    try
    {
        return run_benchmark(argc, argv);
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "benchmark: %s\n", error.what());
        return 1;
    }
}
