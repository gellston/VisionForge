#include "image.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace visionforge::v1
{
namespace
{
std::size_t checked_multiply(std::size_t a, std::size_t b)
{
    if (b != 0 && a > (std::numeric_limits<std::size_t>::max)() / b)
    {
        throw std::length_error("Image size overflow");
    }

    return a * b;
}
}

struct image::impl
{
    std::size_t width;
    std::size_t height;
    std::size_t channel;
    std::size_t stride;
    std::vector<std::uint8_t> pixels;

    impl(std::size_t w, std::size_t h, std::size_t c)
        : width(w),
          height(h),
          channel(c),
          stride(checked_multiply(w, c)),
          pixels(checked_multiply(stride, h), 0)
    {
    }
};

image::image() noexcept = default;

image::image(std::size_t width, std::size_t height, std::size_t channel)
{
    if (width == 0 || height == 0 || channel == 0)
    {
        throw std::invalid_argument(
            "Width, height and channel must be greater than zero");
    }

    impl_ = std::make_unique<impl>(width, height, channel);
}

image::~image() noexcept = default;

image::image(const image& other)
    : impl_(other.impl_ ? std::make_unique<impl>(*other.impl_) : nullptr)
{
}

image& image::operator=(const image& other)
{
    if (this != &other)
    {
        auto copied = other.impl_
                          ? std::make_unique<impl>(*other.impl_)
                          : nullptr;
        impl_.swap(copied);
    }

    return *this;
}

image::image(image&& other) noexcept = default;
image& image::operator=(image&& other) noexcept = default;

std::size_t image::width() const noexcept
{
    return impl_ ? impl_->width : 0;
}

std::size_t image::height() const noexcept
{
    return impl_ ? impl_->height : 0;
}

std::size_t image::channel() const noexcept
{
    return impl_ ? impl_->channel : 0;
}

std::size_t image::stride() const noexcept
{
    return impl_ ? impl_->stride : 0;
}

std::uint8_t* image::data() noexcept
{
    return impl_ ? impl_->pixels.data() : nullptr;
}

const std::uint8_t* image::data() const noexcept
{
    return impl_ ? impl_->pixels.data() : nullptr;
}

image image::resize(std::size_t width, std::size_t height,
                    resize_interpolation interpolation) const
{
    if (!impl_)
        throw std::logic_error("Cannot resize an empty image");
    if (width == 0 || height == 0)
        throw std::invalid_argument("Resize dimensions must be greater than zero");
    if (interpolation != resize_interpolation::nearest
        && interpolation != resize_interpolation::bilinear)
        throw std::invalid_argument("Unknown resize interpolation");
    if (width == impl_->width && height == impl_->height)
        return *this;

    image result(width, height, impl_->channel);
    const double scale_x = static_cast<double>(impl_->width) / static_cast<double>(width);
    const double scale_y = static_cast<double>(impl_->height) / static_cast<double>(height);
    const auto* source = data();
    auto* destination = result.data();
    const auto channels = channel();
    for (std::size_t y = 0; y < height; ++y)
    {
        for (std::size_t x = 0; x < width; ++x)
        {
            const auto output = y * result.stride() + x * channels;
            if (interpolation == resize_interpolation::nearest)
            {
                const auto sx = (std::min)(static_cast<std::size_t>((static_cast<double>(x) + 0.5) * scale_x), impl_->width - 1);
                const auto sy = (std::min)(static_cast<std::size_t>((static_cast<double>(y) + 0.5) * scale_y), impl_->height - 1);
                std::copy_n(source + sy * stride() + sx * channels, channels, destination + output);
                continue;
            }

            const double sx = std::clamp((static_cast<double>(x) + 0.5) * scale_x - 0.5,
                                         0.0, static_cast<double>(impl_->width - 1));
            const double sy = std::clamp((static_cast<double>(y) + 0.5) * scale_y - 0.5,
                                         0.0, static_cast<double>(impl_->height - 1));
            const auto x0 = static_cast<std::size_t>(sx), y0 = static_cast<std::size_t>(sy);
            const auto x1 = (std::min)(x0 + 1, impl_->width - 1);
            const auto y1 = (std::min)(y0 + 1, impl_->height - 1);
            const double fx = sx - static_cast<double>(x0), fy = sy - static_cast<double>(y0);
            for (std::size_t c = 0; c < channels; ++c)
            {
                const double top = (1.0 - fx) * source[y0 * stride() + x0 * channels + c]
                                 + fx * source[y0 * stride() + x1 * channels + c];
                const double bottom = (1.0 - fx) * source[y1 * stride() + x0 * channels + c]
                                    + fx * source[y1 * stride() + x1 * channels + c];
                const double value = (1.0 - fy) * top + fy * bottom;
                destination[output + c] = static_cast<std::uint8_t>(std::clamp(std::lround(value), 0L, 255L));
            }
        }
    }
    return result;
}

image image::crop(std::size_t x, std::size_t y, std::size_t width, std::size_t height) const
{
    if (!impl_)
        throw std::logic_error("Cannot crop an empty image");
    if (width == 0 || height == 0)
        throw std::invalid_argument("Crop dimensions must be greater than zero");
    // Subtract after testing the origin so oversized unsigned arguments cannot wrap.
    if (x >= impl_->width || y >= impl_->height
        || width > impl_->width - x || height > impl_->height - y)
        throw std::out_of_range("Crop rectangle is outside the image");
    image result(width, height, channel());
    for (std::size_t row = 0; row < height; ++row)
        std::copy_n(data() + (y + row) * stride() + x * channel(),
                    result.stride(), result.data() + row * result.stride());
    return result;
}
}
