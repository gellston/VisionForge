#pragma once

#include "shape_matching_export.hpp"
#include "image.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace visionforge::v1
{
struct shape_match_options
{
    // Includes the original image. One level searches the original image directly.
    // Every template level must be at least 5 x 5.
    std::size_t pyramid_levels = 3;
    double rotation_min_degrees = -180.0;
    double rotation_max_degrees = 180.0;
    double rotation_step_degrees = 5.0;
    double minimum_score = 0.7;
    double gradient_threshold = 8.0;
    std::size_t maximum_features = 512;
    std::size_t candidate_count = 32;
    // refine() searches each coarse candidate inside this neighborhood instead of
    // the whole level. Both are counts: radius in pixels, steps in rotation steps.
    // 0 refines at the coarse position/angle only. Cost per candidate is
    // (2 * radius + 1)^2 * (2 * steps + 1) scores.
    std::size_t refine_position_radius = 2;
    std::size_t refine_angle_steps = 1;
    // Summed-area prefilter on the first (coarsest) scan: a candidate whose
    // bounding box holds fewer edge pixels than this is dropped before scoring.
    // 0 keeps the base behaviour of dropping only boxes with no edge at all;
    // raise it to also drop boxes whose edges are too sparse to be worth scoring.
    // This trades recall for speed: too high and real matches get filtered out.
    std::size_t edge_count_threshold = 0;
};

struct shape_match_point
{
    double x = 0.0;
    double y = 0.0;
};

struct shape_match_result
{
    bool found = false;
    // Template center in original source-image pixels; valid when found is true.
    double x = 0.0;
    double y = 0.0;
    // Positive angles rotate clockwise in image coordinates.
    double angle_degrees = 0.0;
    double score = 0.0;
    // Rotated template corners in original source-image pixels; valid when found is true.
    // Names refer to the unrotated template corners, and follow its rotation.
    // Bounds use pixel centers: (0, 0) through (width - 1, height - 1).
    // Draw in order: top_left, top_right, bottom_right, bottom_left, top_left.
    shape_match_point top_left;
    shape_match_point top_right;
    shape_match_point bottom_left;
    shape_match_point bottom_right;
};

// The template's own corner box at one rotation angle, relative to the template
// center rather than any source image. Add a center (for example shape_match_result::x/y)
// to place it on an image.
struct shape_rotation_corners
{
    double angle_degrees = 0.0;
    // Names refer to the unrotated template corners, and follow its rotation.
    // Bounds use pixel centers: (0, 0) through (width - 1, height - 1).
    shape_match_point top_left;
    shape_match_point top_right;
    shape_match_point bottom_left;
    shape_match_point bottom_right;
};

class shape_matcher
{
public:
    using shared_shape_matcher = std::shared_ptr<shape_matcher>;
    using shared_const_shape_matcher = std::shared_ptr<const shape_matcher>;
    using unique_shape_matcher = std::unique_ptr<shape_matcher>;

    visionforge_shape_api explicit shape_matcher(const shape_match_options& options = {});
    visionforge_shape_api ~shape_matcher() noexcept;
    visionforge_shape_api shape_matcher(const shape_matcher& other);
    visionforge_shape_api shape_matcher& operator=(const shape_matcher& other);
    visionforge_shape_api shape_matcher(shape_matcher&& other) noexcept;
    visionforge_shape_api shape_matcher& operator=(shape_matcher&& other) noexcept;

    // Extracts and owns template features; does not retain the input image.
    // Precomputes every rotation's features and corner box once, so match() reuses them.
    visionforge_shape_api void set_template(const image& template_image);
    visionforge_shape_api void clear_template() noexcept;
    [[nodiscard]] visionforge_shape_api bool has_template() const noexcept;
    [[nodiscard]] visionforge_shape_api shape_match_options options() const noexcept;
    [[nodiscard]] visionforge_shape_api shape_match_result match(const image& source) const;

    // One entry per configured rotation angle, in the order match() searches them.
    // Empty when no template is set. Corners are relative to the template center.
    [[nodiscard]] visionforge_shape_api std::vector<shape_rotation_corners> rotated_template_corners() const;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};
}
