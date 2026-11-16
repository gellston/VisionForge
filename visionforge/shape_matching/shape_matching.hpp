#pragma once

#include "shape_matching_export.hpp"
#include "image.hpp"

#include <cstddef>
#include <memory>

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
    visionforge_shape_api void set_template(const image& template_image);
    visionforge_shape_api void clear_template() noexcept;
    [[nodiscard]] visionforge_shape_api bool has_template() const noexcept;
    [[nodiscard]] visionforge_shape_api shape_match_options options() const noexcept;
    [[nodiscard]] visionforge_shape_api shape_match_result match(const image& source) const;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};
}
