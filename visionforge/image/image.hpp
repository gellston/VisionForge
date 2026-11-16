#pragma once

#include "export.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace visionforge::v1
{
enum class resize_interpolation
{
    nearest,
    bilinear
};

class image
{
public:
    using shared_image = std::shared_ptr<image>;
    using shared_const_image = std::shared_ptr<const image>;
    using unique_image = std::unique_ptr<image>;

    visionforge_api image() noexcept;
    visionforge_api image(std::size_t width, std::size_t height, std::size_t channel);
    visionforge_api ~image() noexcept;

    // Copies own independent pixel buffers.
    visionforge_api image(const image& other);
    visionforge_api image& operator=(const image& other);
    visionforge_api image(image&& other) noexcept;
    visionforge_api image& operator=(image&& other) noexcept;

    [[nodiscard]] visionforge_api std::size_t width() const noexcept;
    [[nodiscard]] visionforge_api std::size_t height() const noexcept;
    [[nodiscard]] visionforge_api std::size_t channel() const noexcept;

    // Number of bytes per row; channels are interleaved, one byte each.
    [[nodiscard]] visionforge_api std::size_t stride() const noexcept;

    // Empty and moved-from images return nullptr.
    [[nodiscard]] visionforge_api std::uint8_t* data() noexcept;
    [[nodiscard]] visionforge_api const std::uint8_t* data() const noexcept;

    // Returns an independent image without modifying the source. Channels are unchanged.
    // Bilinear sampling uses pixel centers and replicated borders.
    [[nodiscard]] visionforge_api image resize(
        std::size_t width, std::size_t height,
        resize_interpolation interpolation = resize_interpolation::bilinear) const;

    // Copies [x, x + width) x [y, y + height); the rectangle must be inside the image.
    [[nodiscard]] visionforge_api image crop(
        std::size_t x, std::size_t y, std::size_t width, std::size_t height) const;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};
}
