#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace visionforge::v1
{
class image
{
public:
    using shared_image = std::shared_ptr<image>;
    using shared_const_image = std::shared_ptr<const image>;
    using unique_image = std::unique_ptr<image>;

    image() noexcept;
    image(std::size_t width, std::size_t height, std::size_t channel);
    ~image();

    // Copies own independent pixel buffers.
    image(const image& other);
    image& operator=(const image& other);
    image(image&& other) noexcept;
    image& operator=(image&& other) noexcept;

    [[nodiscard]] std::size_t width() const noexcept;
    [[nodiscard]] std::size_t height() const noexcept;
    [[nodiscard]] std::size_t channel() const noexcept;

    // Number of bytes per row; channels are interleaved, one byte each.
    [[nodiscard]] std::size_t stride() const noexcept;

    // Empty and moved-from images return nullptr.
    [[nodiscard]] std::uint8_t* data() noexcept;
    [[nodiscard]] const std::uint8_t* data() const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};
}
