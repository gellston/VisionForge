#include "image.hpp"

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

image::~image() = default;

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
}
