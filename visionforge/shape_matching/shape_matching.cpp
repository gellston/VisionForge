#include "shape_matching.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace visionforge::v1
{
namespace
{
constexpr double pi = 3.14159265358979323846;

struct plane
{
    int width;
    int height;
    // Pixels are 0..255, and float's 24-bit mantissa represents every integer in
    // that range exactly -- as it does the integer sums the binomial filter
    // produces -- so a single-channel pipeline is bit-identical to double while
    // halving every read and write of these planes. Multi-channel grayscale
    // divides by 3 and does differ in the last bits.
    std::vector<float> pixels;
};

struct gradient
{
    double x = 0.0;
    double y = 0.0;
};

// An inclusive pixel box. refine() only ever touches a handful of small boxes
// around its candidates, so a fine level never needs a full gradient map.
struct region_box
{
    int x0 = 0;
    int y0 = 0;
    int x1 = 0;  // inclusive
    int y1 = 0;
};
struct gradient_map
{
    int width;
    int height;
    std::vector<gradient> pixels;
    // edge[i] != 0 when pixels[i] already meets the threshold used for this map.
    // Lets score() skip bilinear interpolation for cells that cannot contribute.
    std::vector<std::uint8_t> edge;
    // Summed-area table over edge, sized (width + 1) * (height + 1), filled by
    // build_edge_integral(). Empty when not built (the filter then does nothing).
    std::vector<std::uint32_t> integral;
    int integral_stride = 0;
    // Fraction of pixels meeting the threshold. Both fast paths below only pay
    // off on sparse maps: on a dense map the box almost always holds an edge, so
    // the checks run on every candidate and reject nothing.
    double edge_density = 0.0;
    bool use_prefilter = false;   // summed-area skip in full_search
    bool use_edge_mask = false;   // skip interpolation inside score()
};

struct feature
{
    double x;
    double y;
    double nx;
    double ny;
};

// Temporary instrumentation: counts calls in the hot path so the bottleneck can
// be named instead of guessed. Gated by VF_PROFILE in the environment; zero cost
// when unset. Remove together with the counters it guards.
struct profile_counters
{
    std::uint64_t score_calls = 0;
    std::uint64_t fit_fail = 0;          // score() returned early: box outside map
    std::uint64_t early_exit = 0;        // score() hit the cutoff before visiting all
    std::uint64_t features_visited = 0;  // per-feature iterations
    std::uint64_t refine_features = 0;   // subset of visited spent inside refine()
    bool in_refine = false;
    std::uint64_t features_scored = 0;   // actually contributed (magnitude >= threshold)
    std::uint64_t interpolate_calls = 0;
    std::uint64_t full_search_positions = 0;
    std::uint64_t prefilter_reject = 0;   // skipped: bounding box holds no edge pixel
    std::uint64_t edge_pixels = 0;        // edge pixels in the coarsest map
    std::uint64_t map_pixels = 0;         // total pixels in that map
    int prefilter_on = 0;
    int edge_mask_on = 0;
    std::uint64_t integer_box_angles = 0;  // angles whose floor() moved out of the loop
    std::uint64_t refine_positions = 0;
    double ms_pyramid = 0.0;
    double ms_gradients = 0.0;
    double ms_integral = 0.0;
    double ms_full_search = 0.0;
    double ms_refine = 0.0;
    double ms_other = 0.0;
};

profile_counters g_prof;
bool g_prof_enabled = false;
inline bool profile_on()
{
    static const bool on = [] {
        char* value = nullptr;
        std::size_t length = 0;
        const bool set = _dupenv_s(&value, &length, "VF_PROFILE") == 0 && value != nullptr;
        free(value);
        return set;
    }();
    return on;
}

inline void profile_reset()
{
    g_prof = profile_counters{};
}

void profile_report(double total_ms)
{
    if (!profile_on())
        return;
    const auto& p = g_prof;
    std::fprintf(stderr,
        "[VF] total=%.2f ms\n"
        "[VF]   pyramid    %8.2f ms (%.1f%%)\n"
        "[VF]   gradients  %8.2f ms (%.1f%%)\n"
        "[VF]   edge-integ %8.2f ms (%.1f%%)\n"
        "[VF]   full_search%8.2f ms (%.1f%%)  positions=%llu\n"
        "[VF]   refine     %8.2f ms (%.1f%%)  positions=%llu\n"
        "[VF]   other      %8.2f ms (%.1f%%)\n"
        "[VF]   score() calls=%llu  fit_fail=%llu  early_exit=%llu  prefilter_reject=%llu (%.1f%%)\n"
        "[VF]   edge density in coarse map: %llu / %llu px (%.2f%%)\n"
        "[VF]   fast paths: prefilter=%d edge_mask=%d integer_box_angles=%llu\n"
        "[VF]   features visited=%llu  refine=%llu  full_search=%llu  scored=%llu interpolate=%llu\n"
        "[VF]   ns/score=%.1f  ns/visited_feature=%.2f\n",
        total_ms,
        p.ms_pyramid, 100.0 * p.ms_pyramid / total_ms,
        p.ms_gradients, 100.0 * p.ms_gradients / total_ms,
        p.ms_integral, 100.0 * p.ms_integral / total_ms,
        p.ms_full_search, 100.0 * p.ms_full_search / total_ms,
        static_cast<unsigned long long>(p.full_search_positions),
        p.ms_refine, 100.0 * p.ms_refine / total_ms,
        static_cast<unsigned long long>(p.refine_positions),
        p.ms_other, 100.0 * p.ms_other / total_ms,
        static_cast<unsigned long long>(p.score_calls),
        static_cast<unsigned long long>(p.fit_fail),
        static_cast<unsigned long long>(p.early_exit),
        static_cast<unsigned long long>(p.prefilter_reject),
        p.full_search_positions ? 100.0 * static_cast<double>(p.prefilter_reject)
                                           / static_cast<double>(p.full_search_positions) : 0.0,
        static_cast<unsigned long long>(p.edge_pixels),
        static_cast<unsigned long long>(p.map_pixels),
        p.map_pixels ? 100.0 * static_cast<double>(p.edge_pixels)
                                / static_cast<double>(p.map_pixels) : 0.0,
        p.prefilter_on,
        p.edge_mask_on,
        static_cast<unsigned long long>(p.integer_box_angles),
        static_cast<unsigned long long>(p.features_visited),
        static_cast<unsigned long long>(p.refine_features),
        static_cast<unsigned long long>(p.features_visited - p.refine_features),
        static_cast<unsigned long long>(p.features_scored),
        static_cast<unsigned long long>(p.interpolate_calls),
        p.score_calls ? total_ms * 1e6 / static_cast<double>(p.score_calls) : 0.0,
        p.features_visited ? total_ms * 1e6 / static_cast<double>(p.features_visited) : 0.0);
}

struct model_level
{
    double center_x;
    double center_y;
    double half_width;
    double half_height;
    std::vector<feature> features;
};

struct rotated_model
{
    double min_x;
    double max_x;
    double min_y;
    double max_y;
    std::vector<feature> features;
    shape_match_point top_left;
    shape_match_point top_right;
    shape_match_point bottom_left;
    shape_match_point bottom_right;
};

struct candidate
{
    double x;
    double y;
    std::size_t angle;
    double score;
};

std::size_t offset(int x, int y, int width)
{
    return static_cast<std::size_t>(y) * static_cast<std::size_t>(width)
         + static_cast<std::size_t>(x);
}

void validate_options(const shape_match_options& value)
{
    if (value.pyramid_levels == 0 || value.pyramid_levels > 16
        || value.candidate_count == 0 || value.candidate_count > 256
        || value.maximum_features < 4 || value.maximum_features > 4096
        || value.refine_position_radius > 256 || value.refine_angle_steps > 4094
        || value.edge_count_threshold > 1000000
        || !std::isfinite(value.minimum_score) || value.minimum_score <= 0.0
        || value.minimum_score > 1.0
        || !std::isfinite(value.gradient_threshold) || value.gradient_threshold <= 0.0
        || value.gradient_threshold > 255.0
        || !std::isfinite(value.rotation_min_degrees)
        || !std::isfinite(value.rotation_max_degrees)
        || !std::isfinite(value.rotation_step_degrees)
        || value.rotation_min_degrees < -360.0 || value.rotation_max_degrees > 360.0
        || value.rotation_min_degrees > value.rotation_max_degrees
        || value.rotation_max_degrees - value.rotation_min_degrees > 360.0
        || value.rotation_step_degrees <= 0.0)
    {
        throw std::invalid_argument("Invalid shape matching options");
    }
    const double steps = (value.rotation_max_degrees - value.rotation_min_degrees)
                       / value.rotation_step_degrees;
    if (!std::isfinite(steps) || steps > 4094.0)
        throw std::invalid_argument("Rotation range has too many angle candidates");
}

std::vector<double> make_angles(const shape_match_options& options)
{
    const double span = options.rotation_max_degrees - options.rotation_min_degrees;
    const auto steps = static_cast<std::size_t>(std::floor(span / options.rotation_step_degrees));
    std::vector<double> angles;
    for (std::size_t i = 0; i <= steps; ++i)
        angles.push_back(options.rotation_min_degrees + static_cast<double>(i) * options.rotation_step_degrees);
    if (options.rotation_max_degrees - angles.back() > 1e-9)
        angles.push_back(options.rotation_max_degrees);
    // The two endpoints of a complete turn describe the same orientation.
    if (span == 360.0 && angles.size() > 1)
        angles.pop_back();
    return angles;
}

plane grayscale(const image& input)
{
    const auto channels = input.channel();
    const auto max_dimension = static_cast<std::size_t>((std::numeric_limits<int>::max)() - 2);
    if (!input.data() || input.width() < 3 || input.height() < 3
        || input.width() > max_dimension || input.height() > max_dimension
        || (channels != 1 && channels != 3 && channels != 4))
        throw std::invalid_argument("Expected a nonempty 1-, 3- or 4-channel image of at least 3 x 3");

    plane result{static_cast<int>(input.width()), static_cast<int>(input.height()), {}};
    result.pixels.resize(input.width() * input.height());
    const auto* data = input.data();
    for (std::size_t y = 0; y < input.height(); ++y)
    {
        for (std::size_t x = 0; x < input.width(); ++x)
        {
            const auto index = y * input.stride() + x * channels;
            const double value = channels == 1 ? data[index]
                : (static_cast<double>(data[index]) + data[index + 1] + data[index + 2]) / 3.0;
            result.pixels[y * input.width() + x] = static_cast<float>(value);
        }
    }
    return result;
}

// Low-pass filtering here prevents aliasing during pyramid decimation only.
plane downsample(const plane& input)
{
    const int out_w = (input.width + 1) / 2;
    const int out_h = (input.height + 1) / 2;
    plane result{out_w, out_h, {}};
    result.pixels.resize(static_cast<std::size_t>(out_w) * out_h);
    constexpr int w[] = {1, 4, 6, 4, 1};

    // The binomial kernel is an outer product, so summing over dx does not
    // depend on dy: run that half once per source row and reuse it for the
    // five output rows that overlap the row. That turns N*N taps per output
    // into the 2*ceil(src/2) + out_y rows the two passes together visit --
    // N/3 times fewer multiply-adds (1.66x at N=5, and no gain at all below
    // N=5, since a 3-tap pass over every source row costs what it saves).
    //
    // Five slots are enough: an output row reads source rows 2y-2..2y+2, and
    // each source row stays live for only about three output rows, so the next
    // five rows evict nothing still in use.
    constexpr int cache_rows = 5;
    std::vector<float> cache(static_cast<std::size_t>(cache_rows) * out_w);
    int owner[cache_rows];
    for (int i = 0; i < cache_rows; ++i)
        owner[i] = -1;

    // Every entry here is an exact integer for single-channel input: the taps
    // are integers and the running sum stays under 65,280, well inside float's
    // 24-bit mantissa. So storing the intermediate as float costs nothing and
    // the two passes reproduce the single-pass result bit for bit.
    const auto horizontal = [&](int row) -> const float* {
        const int r = std::clamp(row, 0, input.height - 1);
        const int slot = r % cache_rows;
        if (owner[slot] == r)
            return cache.data() + static_cast<std::size_t>(slot) * out_w;
        const float* source = input.pixels.data() + static_cast<std::size_t>(r) * input.width;
        float* destination = cache.data() + static_cast<std::size_t>(slot) * out_w;
        for (int x = 0; x < out_w; ++x)
        {
            float sum = 0.0f;
            for (int dx = -2; dx <= 2; ++dx)
            {
                const int index = 2 * x + dx;
                sum += source[std::clamp(index, 0, input.width - 1)] * w[dx + 2];
            }
            destination[x] = sum;
        }
        owner[slot] = r;
        return destination;
    };

    for (int y = 0; y < out_h; ++y)
    {
        const float* rows[5];
        for (int dy = -2; dy <= 2; ++dy)
            rows[dy + 2] = horizontal(2 * y + dy);
        for (int x = 0; x < out_w; ++x)
        {
            float sum = 0.0f;
            for (int dy = -2; dy <= 2; ++dy)
                sum += rows[dy + 2][x] * w[dy + 2];
            result.pixels[static_cast<std::size_t>(y) * out_w + x] = sum / 256.0f;
        }
    }
    return result;
}

std::vector<plane> pyramid(const image& input, std::size_t levels)
{
    std::vector<plane> result;
    result.push_back(grayscale(input));
    while (result.size() < levels)
        result.push_back(downsample(result.back()));
    return result;
}

gradient_map gradients(const plane& input, double threshold)
{
    gradient_map result{input.width, input.height, {}, {}};
    result.pixels.resize(input.pixels.size());
    result.edge.resize(input.pixels.size());
    const double threshold_squared = threshold * threshold;
    std::uint64_t edge_total = 0;
    for (int y = 1; y < input.height - 1; ++y)
    {
        for (int x = 1; x < input.width - 1; ++x)
        {
            const auto value = [&](int dx, int dy) { return input.pixels[offset(x + dx, y + dy, input.width)]; };
            const gradient g{
                (value(1, -1) + 2.0 * value(1, 0) + value(1, 1)
                 - value(-1, -1) - 2.0 * value(-1, 0) - value(-1, 1)) / 8.0,
                (value(-1, 1) + 2.0 * value(0, 1) + value(1, 1)
                 - value(-1, -1) - 2.0 * value(0, -1) - value(1, -1)) / 8.0};
            result.pixels[offset(x, y, input.width)] = g;
            const auto is_edge = (g.x * g.x + g.y * g.y) >= threshold_squared ? 1 : 0;
            result.edge[offset(x, y, input.width)] = is_edge;
            edge_total += is_edge;
        }
    }
    result.edge_density = result.pixels.empty()
        ? 0.0
        : static_cast<double>(edge_total) / static_cast<double>(result.pixels.size());
    return result;
}

// Build a gradient map for a fine level. score() reads a gradient only at
// (position + rotated feature) and its bilinear neighbours, so everything
// outside `regions` is never touched -- those entries stay zero and cost
// nothing to leave alone. Only the pixels inside a region get a real Sobel.
// No edge mask and no summed-area table are built: on a fine level the map is
// dense in edges, which is exactly where those two fast paths stop paying.
gradient_map lazy_gradients(const plane& input, double threshold,
                            const std::vector<region_box>& regions)
{
    gradient_map result;
    result.width = input.width;
    result.height = input.height;
    result.pixels.assign(static_cast<std::size_t>(input.width) * input.height, gradient{});
    const double threshold_squared = threshold * threshold;
    const auto offset = [](int x, int y, int width) {
        return static_cast<std::size_t>(y) * static_cast<std::size_t>(width)
             + static_cast<std::size_t>(x);
    };
    for (const auto& box : regions)
    {
        // Same bounds gradients() uses: the Sobel taps reach one pixel either
        // side, so the last computable column is width - 2. Clamping to
        // width - 1 here made value(1, 0) read input.pixels[width] -- one past
        // the end, which Debug's checked operator[] rejects with a dialog and
        // Release silently reads as garbage. Border pixels stay zero either
        // way, exactly as gradients() leaves them.
        const int lo_x = std::max(1, box.x0);
        const int hi_x = std::min(input.width - 2, box.x1);
        const int lo_y = std::max(1, box.y0);
        const int hi_y = std::min(input.height - 2, box.y1);
        for (int y = lo_y; y <= hi_y; ++y)
        {
            for (int x = lo_x; x <= hi_x; ++x)
            {
                const auto value = [&](int dx, int dy) { return input.pixels[offset(x + dx, y + dy, input.width)]; };
                const gradient g{
                    (value(1, -1) + 2.0 * value(1, 0) + value(1, 1)
                     - value(-1, -1) - 2.0 * value(-1, 0) - value(-1, 1)) / 8.0,
                    (value(-1, 1) + 2.0 * value(0, 1) + value(1, 1)
                     - value(-1, -1) - 2.0 * value(0, -1) - value(1, -1)) / 8.0};
                result.pixels[offset(x, y, input.width)] = g;
            }
        }
    }
    // Dense-in-edges: both fast paths are off, so no edge mask or integral.
    result.use_prefilter = false;
    result.use_edge_mask = false;
    return result;
}

// The boxes refine() will read at the finer level, derived from the candidate
// positions it is about to expand. score() samples at x + feature.x where
// x = parent.x * 2 + dx and dx spans +-radius, then reads floor() and
// floor()+1 of that for the bilinear blend -- so the box has to cover the
// position range, the rotated feature extent, and one extra pixel.
std::vector<region_box> visit_regions(const std::vector<candidate>& parents,
                                      const std::vector<rotated_model>& rotated,
                                      const shape_match_options& settings)
{
    int min_fx = std::numeric_limits<int>::max();
    int min_fy = std::numeric_limits<int>::max();
    int max_fx = std::numeric_limits<int>::min();
    int max_fy = std::numeric_limits<int>::min();
    for (const auto& model : rotated)
    {
        min_fx = std::min(min_fx, static_cast<int>(std::floor(model.min_x)));
        min_fy = std::min(min_fy, static_cast<int>(std::floor(model.min_y)));
        max_fx = std::max(max_fx, static_cast<int>(std::floor(model.max_x)));
        max_fy = std::max(max_fy, static_cast<int>(std::floor(model.max_y)));
    }
    const int radius = static_cast<int>(settings.refine_position_radius);  // validated <= 256
    std::vector<region_box> regions;
    regions.reserve(parents.size());
    for (const auto& parent : parents)
    {
        const int cx = static_cast<int>(std::lround(parent.x * 2.0));
        const int cy = static_cast<int>(std::lround(parent.y * 2.0));
        regions.push_back({cx - radius + min_fx - 1, cy - radius + min_fy - 1,
                           cx + radius + max_fx + 1, cy + radius + max_fy + 1});
    }
    return regions;
}

// Turn the two fast paths on or off for this map. A candidate box that on
// average holds fewer than a couple of edges can often be empty, which is
// exactly what the summed-area skip exploits; when the map is dense that check
// costs reads and rejects nothing. Likewise, skipping interpolation only wins if
// most samples land on a non-edge, which a dense map cannot offer.
void configure_edge_paths(gradient_map& map, const model_level& model,
                          std::size_t edge_count_threshold)
{
    const double box_area = (2.0 * model.half_width + 2.0) * (2.0 * model.half_height + 2.0);
    // An explicitly set threshold is a request from the caller and always wins
    // over the density heuristic; the heuristic only guards the default (0),
    // where an empty-box skip would otherwise be pure overhead on a dense map.
    map.use_prefilter = edge_count_threshold > 0 || map.edge_density * box_area < 4.0;
    map.use_edge_mask = map.edge_density < 0.05;
}

// Summed-area table over the edge mask: integral[(y+1)*stride + (x+1)] holds the
// number of edge pixels in [0,x] x [0,y]. A rectangle count is then four reads.
void build_edge_integral(gradient_map& map)
{
    map.integral_stride = map.width + 1;
    map.integral.assign(static_cast<std::size_t>(map.integral_stride)
                            * static_cast<std::size_t>(map.height + 1), 0u);
    for (int y = 0; y < map.height; ++y)
    {
        const auto* above = map.integral.data() + static_cast<std::size_t>(y) * map.integral_stride;
        auto* row = map.integral.data() + static_cast<std::size_t>(y + 1) * map.integral_stride;
        const auto* edge_row = map.edge.data() + static_cast<std::size_t>(y) * map.width;
        std::uint32_t running = 0;
        for (int x = 0; x < map.width; ++x)
        {
            running += edge_row[x];
            row[x + 1] = above[x + 1] + running;
        }
    }
}

std::uint32_t edge_count(const gradient_map& map, int x0, int y0, int x1, int y1)
{
    const auto base = map.integral.data();
    const auto stride = map.integral_stride;
    const auto at = [&](int x, int y) { return base[static_cast<std::size_t>(y) * stride + x]; };
    return at(x1 + 1, y1 + 1) - at(x0, y1 + 1) - at(x1 + 1, y0) + at(x0, y0);
}

// Should score() run at this candidate center at all?
//
// Every feature sits inside the rotated template's bounding box, and a feature
// only contributes when one of its four bilinear corners is an edge pixel. So if
// that box holds too few edge pixels, every feature is worth (almost) nothing,
// score() cannot reach the candidate list, and full_search can skip the call.
//
// edge_count_threshold == 0 (the default) needs at least one edge pixel, which
// is the provable case: no edge at all means every feature scores exactly zero,
// so the skip changes nothing. Raising the threshold additionally drops boxes
// that do have edges but too few to matter -- that is a deliberate recall
// trade-off, not a proof, and it can hide a real match.

model_level extract(const plane& input, double center_x, double center_y,
                    const shape_match_options& options)
{
    model_level result{center_x, center_y, center_x, center_y, {}};
    const auto map = gradients(input, options.gradient_threshold);
    std::vector<feature> all;
    for (int y = 1; y < input.height - 1; ++y)
    {
        for (int x = 1; x < input.width - 1; ++x)
        {
            const auto& g = map.pixels[offset(x, y, input.width)];
            const double magnitude = std::hypot(g.x, g.y);
            if (magnitude >= options.gradient_threshold)
                all.push_back({x - center_x, y - center_y, g.x / magnitude, g.y / magnitude});
        }
    }
    if (all.size() < 4)
        throw std::invalid_argument("Template has fewer than four gradient features at a pyramid level");
    const auto count = (std::min)(all.size(), options.maximum_features);
    result.features.reserve(count);
    // Deterministic spatial sampling prevents a dense edge from monopolizing the model.
    for (std::size_t i = 0; i < count; ++i)
        result.features.push_back(all[i * (all.size() / count) + i * (all.size() % count) / count]);
    return result;
}

std::vector<rotated_model> rotations(const model_level& model, const std::vector<double>& angles)
{
    std::vector<rotated_model> result;
    result.reserve(angles.size());
    for (const double angle : angles)
    {
        const double radians = angle * pi / 180.0;
        const double c = std::cos(radians), s = std::sin(radians);
        const double extent_x = std::abs(c) * model.half_width + std::abs(s) * model.half_height;
        const double extent_y = std::abs(s) * model.half_width + std::abs(c) * model.half_height;
        rotated_model rotated{-extent_x, extent_x, -extent_y, extent_y, {}};
        const auto rotate_corner = [&](double x, double y) -> shape_match_point {
            return {c * x - s * y, s * x + c * y};
        };
        rotated.top_left = rotate_corner(-model.half_width, -model.half_height);
        rotated.top_right = rotate_corner(model.half_width, -model.half_height);
        rotated.bottom_left = rotate_corner(-model.half_width, model.half_height);
        rotated.bottom_right = rotate_corner(model.half_width, model.half_height);
        rotated.features.reserve(model.features.size());
        for (const auto& f : model.features)
            rotated.features.push_back({c * f.x - s * f.y, s * f.x + c * f.y,
                                        c * f.nx - s * f.ny, s * f.nx + c * f.ny});
        result.push_back(std::move(rotated));
    }
    return result;
}

bool fits(const gradient_map& map, const rotated_model& model, double x, double y)
{
    constexpr double epsilon = 1e-9;
    return x + model.min_x >= -epsilon && y + model.min_y >= -epsilon
        && x + model.max_x <= map.width - 1 + epsilon
        && y + model.max_y <= map.height - 1 + epsilon;
}

gradient interpolate(const gradient_map& map, double x, double y)
{
    x = std::clamp(x, 0.0, static_cast<double>(map.width - 1));
    y = std::clamp(y, 0.0, static_cast<double>(map.height - 1));
    const int x0 = static_cast<int>(x), y0 = static_cast<int>(y);
    const int x1 = (std::min)(x0 + 1, map.width - 1), y1 = (std::min)(y0 + 1, map.height - 1);
    const double fx = x - x0, fy = y - y0;
    const auto& a = map.pixels[offset(x0, y0, map.width)];
    const auto& b = map.pixels[offset(x1, y0, map.width)];
    const auto& c = map.pixels[offset(x0, y1, map.width)];
    const auto& d = map.pixels[offset(x1, y1, map.width)];
    return {(1.0 - fy) * ((1.0 - fx) * a.x + fx * b.x) + fy * ((1.0 - fx) * c.x + fx * d.x),
            (1.0 - fy) * ((1.0 - fx) * a.y + fx * b.y) + fy * ((1.0 - fx) * c.y + fx * d.y)};
}

// True when at least one of the four bilinear corners is already an edge.
// Interpolation is a convex combination of those four gradients, so if none of
// them reaches the threshold the interpolated magnitude cannot either (triangle
// inequality) and the feature contributes exactly zero. Returns false then, and
// the caller skips the interpolation and the sqrt entirely.
bool touches_edge(const gradient_map& map, double x, double y)
{
    if (!map.use_edge_mask)
        return true;  // dense map: most samples touch an edge, so just interpolate
    x = std::clamp(x, 0.0, static_cast<double>(map.width - 1));
    y = std::clamp(y, 0.0, static_cast<double>(map.height - 1));
    const int x0 = static_cast<int>(x), y0 = static_cast<int>(y);
    const int x1 = (std::min)(x0 + 1, map.width - 1), y1 = (std::min)(y0 + 1, map.height - 1);
    const auto& edge = map.edge;
    return (edge[offset(x0, y0, map.width)] | edge[offset(x1, y0, map.width)]
          | edge[offset(x0, y1, map.width)] | edge[offset(x1, y1, map.width)]) != 0;
}

double score(const gradient_map& map, const rotated_model& model, double x, double y,
             double threshold, double cutoff)
{
    if (profile_on())
        ++g_prof.score_calls;
    if (!fits(map, model, x, y))
    {
        if (profile_on())
            ++g_prof.fit_fail;
        return 0.0;
    }
    double sum = 0.0;
    const double count = static_cast<double>(model.features.size());
    std::size_t visited = 0;
    for (const auto& f : model.features)
    {
        ++visited;
        if (profile_on())
        {
            ++g_prof.features_visited;
            if (g_prof.in_refine)
                ++g_prof.refine_features;
        }
        const double sample_x = x + f.x, sample_y = y + f.y;
        if (!touches_edge(map, sample_x, sample_y))
        {
            // Convex-combination bound: this feature contributes exactly zero.
            if (sum + count - static_cast<double>(visited) < cutoff * count)
            {
                if (profile_on())
                    ++g_prof.early_exit;
                return 0.0;
            }
            continue;
        }
        const auto g = interpolate(map, sample_x, sample_y);
        if (profile_on())
            ++g_prof.interpolate_calls;
        // Compare squared magnitudes: Sobel output is bounded well inside double
        // range here, so this is equivalent to comparing std::hypot against the
        // threshold while skipping the sqrt for features that contribute nothing.
        const double magnitude_squared = g.x * g.x + g.y * g.y;
        if (magnitude_squared >= threshold * threshold)
        {
            sum += std::clamp((f.nx * g.x + f.ny * g.y) / std::sqrt(magnitude_squared), 0.0, 1.0);
            if (profile_on())
                ++g_prof.features_scored;
        }
        // Even perfect remaining features cannot enter the candidate list.
        if (sum + count - static_cast<double>(visited) < cutoff * count)
        {
            if (profile_on())
                ++g_prof.early_exit;
            return 0.0;
        }
    }
    // Missing or opposite-polarity edges contribute zero, but remain in the denominator.
    return std::clamp(sum / count, 0.0, 1.0);
}

void retain(std::vector<candidate>& list, const candidate& value, std::size_t capacity)
{
    if (value.score <= 0.0 || (list.size() == capacity && value.score <= list.back().score))
        return;
    for (auto it = list.begin(); it != list.end(); ++it)
    {
        const auto angle_distance = it->angle > value.angle ? it->angle - value.angle : value.angle - it->angle;
        if (angle_distance <= 1 && std::hypot(it->x - value.x, it->y - value.y) <= 2.0)
        {
            if (it->score >= value.score)
                return;
            list.erase(it);
            break;
        }
    }
    const auto where = std::lower_bound(list.begin(), list.end(), value,
        [](const candidate& a, const candidate& b) { return a.score > b.score; });
    list.insert(where, value);
    if (list.size() > capacity)
        list.pop_back();
}

double cutoff(const std::vector<candidate>& list, std::size_t capacity)
{
    return list.size() == capacity ? list.back().score : 0.0;
}

std::vector<candidate> full_search(const gradient_map& map, const model_level& model,
                                 const std::vector<rotated_model>& rotated,
                                 const shape_match_options& options, std::size_t capacity)
{
    std::vector<candidate> result;
    // Resolved once: every one of these used to be re-read per candidate, and
    // the profile flag alone cost a call on all 22.9M positions.
    const bool prof = profile_on();
    const bool filter = map.use_prefilter && !map.integral.empty();
    const auto* integral = map.integral.data();
    const int stride = map.integral_stride;
    const int last_col = map.width - 1;
    const int last_row = map.height - 1;
    const auto required = (std::max)(options.edge_count_threshold, std::size_t{1});
    for (std::size_t angle = 0; angle < rotated.size(); ++angle)
    {
        const auto& rotation = rotated[angle];
        if (rotation.max_x - rotation.min_x > map.width - 1 + 1e-9
            || rotation.max_y - rotation.min_y > map.height - 1 + 1e-9)
            continue;
        const int min_x = static_cast<int>(std::ceil(-rotation.min_x - model.center_x - 1e-9));
        const int min_y = static_cast<int>(std::ceil(-rotation.min_y - model.center_y - 1e-9));
        const int max_x = static_cast<int>(std::floor(map.width - 1 - rotation.max_x - model.center_x + 1e-9));
        const int max_y = static_cast<int>(std::floor(map.height - 1 - rotation.max_y - model.center_y + 1e-9));

        // floor(cx + c) with cx = x + center_x and integer x equals x + floor(K)
        // for K = center_x + c -- but only if rounding fl(x + K) never carries
        // across an integer, which would shift the box by one pixel. The
        // fractional part of x + K is the same for every x, and ulp grows with
        // |x + K|, so a carry can only start at a binade boundary. Both ends of
        // the range bracket every such boundary: if they agree on the delta, the
        // delta holds for the whole range and floor() leaves the inner loop; if
        // they disagree the original expression is kept. Either way the numbers
        // coming out are identical to what the plain expression would produce.
        int delta_min = 0;
        int delta_max = 0;
        bool integer_box = false;
        if (filter)
        {
            const auto delta_at = [&](double c, int x_at) {
                return static_cast<int>(std::floor((x_at + model.center_x) + c)) - x_at;
            };
            delta_min = delta_at(rotation.min_x, min_x);
            delta_max = delta_at(rotation.max_x, min_x);
            integer_box = delta_min == delta_at(rotation.min_x, max_x)
                       && delta_max == delta_at(rotation.max_x, max_x);
            if (prof && integer_box)
                ++g_prof.integer_box_angles;
        }

        for (int y = min_y; y <= max_y; ++y)
        {
            // Row-invariant half of the prefilter. Written with exactly the same
            // expressions as before, so the arithmetic -- rounding included -- is
            // unchanged; it just runs once per row instead of once per pixel.
            const double cy = y + model.center_y;
            int row0 = static_cast<int>(std::floor(cy + rotation.min_y));
            int row1 = static_cast<int>(std::floor(cy + rotation.max_y)) + 1;
            row0 = (std::max)(row0, 0);
            row1 = (std::min)(row1, last_row);
            if (row0 > row1)
                continue;  // unreachable: the loop bounds keep the template inside
            const auto row_lo = static_cast<std::size_t>(row0) * stride;
            const auto row_hi = static_cast<std::size_t>(row1 + 1) * stride;
            for (int x = min_x; x <= max_x; ++x)
            {
                if (prof)
                    ++g_prof.full_search_positions;
                if (filter)
                {
                    int x0;
                    int x1;
                    if (integer_box)
                    {
                        x0 = x + delta_min;
                        x1 = x + delta_max + 1;
                    }
                    else
                    {
                        const double fallback_cx = x + model.center_x;
                        x0 = static_cast<int>(std::floor(fallback_cx + rotation.min_x));
                        x1 = static_cast<int>(std::floor(fallback_cx + rotation.max_x)) + 1;
                    }
                    x0 = (std::max)(x0, 0);
                    x1 = (std::min)(x1, last_col);
                    if (x0 <= x1)
                    {
                        const auto col_lo = static_cast<std::size_t>(x0);
                        const auto col_hi = static_cast<std::size_t>(x1 + 1);
                        const auto count = integral[row_hi + col_hi] - integral[row_hi + col_lo]
                                         - integral[row_lo + col_hi] + integral[row_lo + col_lo];
                        if (count < required)
                        {
                            if (prof)
                                ++g_prof.prefilter_reject;
                            continue;
                        }
                    }
                    // x0 > x1 (degenerate): worth scoring, exactly as before.
                }
                // Only candidates that survive the filter pay for the rest.
                const double cx = x + model.center_x;
                const double position_cutoff = cutoff(result, capacity);
                retain(result, {cx, cy, angle, score(map, rotation, cx, cy,
                    options.gradient_threshold, position_cutoff)}, capacity);
            }
        }
    }
    return result;
}

std::vector<candidate> refine(const gradient_map& map, const std::vector<rotated_model>& rotated,
                              const std::vector<candidate>& coarse, const shape_match_options& options)
{
    std::vector<candidate> result;
    if (coarse.empty())
        return result;
    const auto angle_radius = options.refine_angle_steps;
    const auto position_radius = static_cast<int>(options.refine_position_radius);
    for (const auto& parent : coarse)
    {
        const auto begin = parent.angle > angle_radius ? parent.angle - angle_radius : 0;
        const auto end = (std::min)(parent.angle + angle_radius, rotated.size() - 1);
        for (auto angle = begin; angle <= end; ++angle)
            for (int dy = -position_radius; dy <= position_radius; ++dy)
                for (int dx = -position_radius; dx <= position_radius; ++dx)
                {
                    if (profile_on())
                        ++g_prof.refine_positions;
                    const double x = parent.x * 2.0 + dx, y = parent.y * 2.0 + dy;
                    retain(result, {x, y, angle, score(map, rotated[angle], x, y,
                        options.gradient_threshold, cutoff(result, options.candidate_count))}, options.candidate_count);
                }
    }
    return result;
}
}

struct shape_matcher::impl
{
    shape_match_options settings;
    std::vector<double> angles;
    std::vector<model_level> model;
    // Rotated features and corner boxes, precomputed by set_template:
    // [pyramid level][angle index]. match() and rotated_template_corners() reuse
    // this instead of rebuilding every rotation on each call.
    std::vector<std::vector<rotated_model>> rotated_levels;

    explicit impl(const shape_match_options& options) : settings(options)
    {
        validate_options(settings);
        angles = make_angles(settings);
    }
};

shape_matcher::shape_matcher(const shape_match_options& options)
    : impl_(std::make_unique<impl>(options)) {}
shape_matcher::~shape_matcher() noexcept = default;
shape_matcher::shape_matcher(const shape_matcher& other)
    : impl_(other.impl_ ? std::make_unique<impl>(*other.impl_) : nullptr) {}
shape_matcher& shape_matcher::operator=(const shape_matcher& other)
{
    if (this != &other)
    {
        auto copied = other.impl_ ? std::make_unique<impl>(*other.impl_) : nullptr;
        impl_.swap(copied);
    }
    return *this;
}
shape_matcher::shape_matcher(shape_matcher&& other) noexcept = default;
shape_matcher& shape_matcher::operator=(shape_matcher&& other) noexcept = default;

void shape_matcher::set_template(const image& template_image)
{
    if (!impl_)
        impl_ = std::make_unique<impl>(shape_match_options{});
    const auto level_count = impl_->settings.pyramid_levels;
    const auto levels = pyramid(template_image, level_count);
    std::vector<model_level> model;
    std::vector<std::vector<rotated_model>> rotated_levels;
    double scale = 1.0;
    for (const auto& level : levels)
    {
        if (level.width < 5 || level.height < 5)
            throw std::invalid_argument("Template pyramid level must be at least 5 x 5; reduce pyramid_levels");
        model.push_back(extract(level, (static_cast<double>(template_image.width()) - 1.0) / (2.0 * scale),
                                (static_cast<double>(template_image.height()) - 1.0) / (2.0 * scale), impl_->settings));
        scale *= 2.0;
    }
    // Build every rotation up front: options (and therefore angles) cannot change
    // after construction, so match() can reuse these for the matcher's lifetime.
    rotated_levels.reserve(model.size());
    for (const auto& level : model)
        rotated_levels.push_back(rotations(level, impl_->angles));
    impl_->model.swap(model);
    impl_->rotated_levels.swap(rotated_levels);
}

void shape_matcher::clear_template() noexcept
{
    if (impl_)
    {
        impl_->model.clear();
        impl_->rotated_levels.clear();
    }
}
bool shape_matcher::has_template() const noexcept { return impl_ && !impl_->model.empty(); }
shape_match_options shape_matcher::options() const noexcept
{
    return impl_ ? impl_->settings : shape_match_options{};
}

std::vector<shape_rotation_corners> shape_matcher::rotated_template_corners() const
{
    std::vector<shape_rotation_corners> result;
    if (!impl_ || impl_->rotated_levels.empty())
        return result;
    // Level 0 is the original template resolution, which callers draw against.
    const auto& level = impl_->rotated_levels.front();
    result.reserve(level.size());
    for (std::size_t i = 0; i < level.size(); ++i)
        result.push_back({impl_->angles[i], level[i].top_left, level[i].top_right,
                          level[i].bottom_left, level[i].bottom_right});
    return result;
}

shape_match_result shape_matcher::match(const image& source) const
{
    if (!has_template())
        throw std::logic_error("Set a template before matching");
    if (impl_->rotated_levels.size() != impl_->model.size())
        throw std::logic_error("Template rotation cache is missing; set the template again");
    const bool prof = profile_on();
    profile_reset();
    using profile_clock = std::chrono::steady_clock;
    const auto prof_now = [] { return profile_clock::now(); };
    const auto prof_ms = [](profile_clock::time_point a, profile_clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    const auto t_total = prof_now();
    auto t_mark = prof_now();
    const auto& settings = impl_->settings;
    const auto count = settings.pyramid_levels;
    const auto levels = pyramid(source, count);
    const auto first = count - 1;
    if (prof) { const auto t = prof_now(); g_prof.ms_pyramid = prof_ms(t_mark, t); t_mark = t; }
    auto map = gradients(levels[first], settings.gradient_threshold);
    if (prof) { const auto t = prof_now(); g_prof.ms_gradients = prof_ms(t_mark, t); t_mark = t; }
    // Decide whether the two fast paths are worth their overhead on this map,
    // then build the summed-area table only if the prefilter will actually run.
    configure_edge_paths(map, impl_->model[first], settings.edge_count_threshold);
    if (prof)
    {
        g_prof.prefilter_on = map.use_prefilter ? 1 : 0;
        g_prof.edge_mask_on = map.use_edge_mask ? 1 : 0;
        g_prof.map_pixels = map.edge.size();
        g_prof.edge_pixels = 0;
        for (const auto e : map.edge)
            g_prof.edge_pixels += e != 0;
    }
    if (prof) { const auto t = prof_now(); t_mark = t; g_prof.ms_integral = 0.0; }
    if (map.use_prefilter)
        build_edge_integral(map);
    if (prof) { const auto t = prof_now(); g_prof.ms_integral = prof_ms(t_mark, t); t_mark = t; }
    // Rotations were precomputed by set_template(); only the source side is rebuilt here.
    const auto* rotated = &impl_->rotated_levels[first];
    auto candidates = full_search(map, impl_->model[first], *rotated, settings,
                                 count > 1 ? settings.candidate_count : 1);
    if (prof) { const auto t = prof_now(); g_prof.ms_full_search = prof_ms(t_mark, t); t_mark = t; }
    for (std::size_t level = first; level > 0 && !candidates.empty(); --level)
    {
        rotated = &impl_->rotated_levels[level - 1];
        // Fine levels are only ever read at (candidate + rotated feature), so
        // build a Sobel for exactly those boxes instead of the whole level.
        // lazy_gradients() leaves both fast paths off for such a map.
        map = lazy_gradients(levels[level - 1], settings.gradient_threshold,
                             visit_regions(candidates, *rotated, settings));
        if (prof) { const auto t = prof_now(); g_prof.ms_gradients += prof_ms(t_mark, t); t_mark = t; }
        g_prof.in_refine = true;
        candidates = refine(map, *rotated, candidates, settings);
        g_prof.in_refine = false;
        if (prof) { const auto t = prof_now(); g_prof.ms_refine += prof_ms(t_mark, t); t_mark = t; }
    }
    if (prof)
    {
        const auto t = prof_now();
        g_prof.ms_other = prof_ms(t_mark, t);
        profile_report(prof_ms(t_total, t));
    }
    if (candidates.empty())
        return {};
    const auto& best = candidates.front();
    const auto& bounds = (*rotated)[best.angle];
    const auto translate = [&](const shape_match_point& point) -> shape_match_point {
        return {best.x + point.x, best.y + point.y};
    };
    return {best.score >= settings.minimum_score, best.x, best.y, impl_->angles[best.angle], best.score,
            translate(bounds.top_left), translate(bounds.top_right),
            translate(bounds.bottom_left), translate(bounds.bottom_right)};
}
}
