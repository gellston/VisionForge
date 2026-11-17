#include "shape_matching.hpp"

#include <algorithm>
#include <cmath>
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
    std::vector<double> pixels;
};

struct gradient
{
    double x = 0.0;
    double y = 0.0;
};

struct gradient_map
{
    int width;
    int height;
    std::vector<gradient> pixels;
};

struct feature
{
    double x;
    double y;
    double nx;
    double ny;
};

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
            result.pixels[y * input.width() + x] = value;
        }
    }
    return result;
}

// Low-pass filtering here prevents aliasing during pyramid decimation only.
plane downsample(const plane& input)
{
    plane result{(input.width + 1) / 2, (input.height + 1) / 2, {}};
    result.pixels.resize(static_cast<std::size_t>(result.width) * result.height);
    constexpr int weights[] = {1, 4, 6, 4, 1};
    for (int y = 0; y < result.height; ++y)
    {
        for (int x = 0; x < result.width; ++x)
        {
            double sum = 0.0;
            for (int dy = -2; dy <= 2; ++dy)
                for (int dx = -2; dx <= 2; ++dx)
                    sum += input.pixels[offset(std::clamp(2 * x + dx, 0, input.width - 1),
                                                std::clamp(2 * y + dy, 0, input.height - 1), input.width)]
                         * weights[dx + 2] * weights[dy + 2];
            result.pixels[offset(x, y, result.width)] = sum / 256.0;
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

gradient_map gradients(const plane& input)
{
    gradient_map result{input.width, input.height, {}};
    result.pixels.resize(input.pixels.size());
    for (int y = 1; y < input.height - 1; ++y)
    {
        for (int x = 1; x < input.width - 1; ++x)
        {
            const auto value = [&](int dx, int dy) { return input.pixels[offset(x + dx, y + dy, input.width)]; };
            result.pixels[offset(x, y, input.width)] = {
                (value(1, -1) + 2.0 * value(1, 0) + value(1, 1)
                 - value(-1, -1) - 2.0 * value(-1, 0) - value(-1, 1)) / 8.0,
                (value(-1, 1) + 2.0 * value(0, 1) + value(1, 1)
                 - value(-1, -1) - 2.0 * value(0, -1) - value(1, -1)) / 8.0};
        }
    }
    return result;
}

model_level extract(const plane& input, double center_x, double center_y,
                    const shape_match_options& options)
{
    model_level result{center_x, center_y, center_x, center_y, {}};
    const auto map = gradients(input);
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

double score(const gradient_map& map, const rotated_model& model, double x, double y,
             double threshold, double cutoff)
{
    if (!fits(map, model, x, y))
        return 0.0;
    double sum = 0.0;
    const double count = static_cast<double>(model.features.size());
    std::size_t visited = 0;
    for (const auto& f : model.features)
    {
        const auto g = interpolate(map, x + f.x, y + f.y);
        const double magnitude = std::hypot(g.x, g.y);
        if (magnitude >= threshold)
            sum += std::clamp((f.nx * g.x + f.ny * g.y) / magnitude, 0.0, 1.0);
        ++visited;
        // Even perfect remaining features cannot enter the candidate list.
        if (sum + count - static_cast<double>(visited) < cutoff * count)
            return 0.0;
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
        for (int y = min_y; y <= max_y; ++y)
            for (int x = min_x; x <= max_x; ++x)
            {
                const double cx = x + model.center_x, cy = y + model.center_y;
                retain(result, {cx, cy, angle, score(map, rotation, cx, cy,
                    options.gradient_threshold, cutoff(result, capacity))}, capacity);
            }
    }
    return result;
}

std::vector<candidate> refine(const gradient_map& map, const std::vector<rotated_model>& rotated,
                              const std::vector<candidate>& coarse, const shape_match_options& options)
{
    std::vector<candidate> result;
    for (const auto& parent : coarse)
    {
        const auto begin = parent.angle == 0 ? 0 : parent.angle - 1;
        const auto end = (std::min)(parent.angle + 1, rotated.size() - 1);
        for (auto angle = begin; angle <= end; ++angle)
            for (int dy = -2; dy <= 2; ++dy)
                for (int dx = -2; dx <= 2; ++dx)
                {
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
    double scale = 1.0;
    for (const auto& level : levels)
    {
        if (level.width < 5 || level.height < 5)
            throw std::invalid_argument("Template pyramid level must be at least 5 x 5; reduce pyramid_levels");
        model.push_back(extract(level, (static_cast<double>(template_image.width()) - 1.0) / (2.0 * scale),
                                (static_cast<double>(template_image.height()) - 1.0) / (2.0 * scale), impl_->settings));
        scale *= 2.0;
    }
    impl_->model.swap(model);
}

void shape_matcher::clear_template() noexcept
{
    if (impl_)
        impl_->model.clear();
}
bool shape_matcher::has_template() const noexcept { return impl_ && !impl_->model.empty(); }
shape_match_options shape_matcher::options() const noexcept
{
    return impl_ ? impl_->settings : shape_match_options{};
}

shape_match_result shape_matcher::match(const image& source) const
{
    if (!has_template())
        throw std::logic_error("Set a template before matching");
    const auto& settings = impl_->settings;
    const auto count = settings.pyramid_levels;
    const auto levels = pyramid(source, count);
    const auto first = count - 1;
    auto map = gradients(levels[first]);
    auto rotated = rotations(impl_->model[first], impl_->angles);
    auto candidates = full_search(map, impl_->model[first], rotated, settings,
                                 count > 1 ? settings.candidate_count : 1);
    for (std::size_t level = first; level > 0 && !candidates.empty(); --level)
    {
        map = gradients(levels[level - 1]);
        rotated = rotations(impl_->model[level - 1], impl_->angles);
        candidates = refine(map, rotated, candidates, settings);
    }
    if (candidates.empty())
        return {};
    const auto& best = candidates.front();
    const auto& bounds = rotated[best.angle];
    const auto translate = [&](const shape_match_point& point) -> shape_match_point {
        return {best.x + point.x, best.y + point.y};
    };
    return {best.score >= settings.minimum_score, best.x, best.y, impl_->angles[best.angle], best.score,
            translate(bounds.top_left), translate(bounds.top_right),
            translate(bounds.bottom_left), translate(bounds.bottom_right)};
}
}
