#include "image.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace
{
using visionforge::v1::image;

static_assert(std::is_same_v<image::shared_image, std::shared_ptr<image>>);
static_assert(std::is_same_v<image::shared_const_image, std::shared_ptr<const image>>);
static_assert(std::is_same_v<image::unique_image, std::unique_ptr<image>>);
static_assert(std::is_same_v<decltype(std::declval<const image&>().data()),
                             const std::uint8_t*>);
static_assert(std::is_nothrow_move_constructible_v<image>);
static_assert(std::is_nothrow_move_assignable_v<image>);

// Release builds must run these checks too.
void require(bool condition)
{
    if (!condition)
        throw std::runtime_error("Image check failed");
}

void require_empty(const image& value)
{
    require(value.width() == 0 && value.height() == 0 && value.channel() == 0);
    require(value.stride() == 0 && value.data() == nullptr);
}

template <typename exception, typename action>
void require_throws(action&& operation)
{
    try { operation(); }
    catch (const exception&) { return; }
    throw std::runtime_error("Expected exception was not thrown");
}

void test_resize()
{
    using visionforge::v1::resize_interpolation;
    image original(2, 2, 1);
    original.data()[0] = 0;
    original.data()[1] = 100;
    original.data()[2] = 150;
    original.data()[3] = 250;
    const image& input = original;
    auto average = input.resize(1, 1);
    require(average.width() == 1 && average.height() == 1 && average.channel() == 1);
    require(average.data()[0] == 125);
    auto nearest = input.resize(4, 4, resize_interpolation::nearest);
    for (std::size_t y = 0; y < 4; ++y)
        for (std::size_t x = 0; x < 4; ++x)
            require(nearest.data()[y * nearest.stride() + x] == original.data()[(y / 2) * 2 + x / 2]);
    auto linear = input.resize(4, 4);
    require(linear.data()[0] == 0 && linear.data()[3] == 100);
    require(linear.data()[12] == 150 && linear.data()[15] == 250);
    require(linear.data()[5] == 63 && linear.data()[10] == 188);
    auto identity = input.resize(2, 2);
    require(identity.data() != original.data());
    identity.data()[0] = 255;
    require(original.data()[0] == 0 && original.width() == 2 && original.height() == 2);

    image color(2, 1, 3);
    for (std::size_t c = 0; c < 3; ++c)
    {
        color.data()[c] = static_cast<std::uint8_t>(c * 10);
        color.data()[3 + c] = static_cast<std::uint8_t>(100 + c * 10);
    }
    const auto mixed = color.resize(1, 1);
    require(mixed.channel() == 3 && mixed.stride() == 3);
    for (std::size_t c = 0; c < 3; ++c)
        require(mixed.data()[c] == 50 + c * 10);
    image single(1, 1, 4);
    const std::uint8_t channels[] = {255, 0, 128, 67};
    for (std::size_t c = 0; c < 4; ++c)
        single.data()[c] = channels[c];
    const auto expanded = single.resize(7, 3);
    for (std::size_t i = 0; i < expanded.stride() * expanded.height(); ++i)
        require(expanded.data()[i] == channels[i % 4]);

    require_throws<std::invalid_argument>([&] { (void)input.resize(0, 1); });
    require_throws<std::invalid_argument>([&] { (void)input.resize(1, 0); });
    require_throws<std::invalid_argument>([&] { (void)input.resize(2, 2, static_cast<resize_interpolation>(99)); });
    require_throws<std::length_error>([&] { (void)input.resize((std::numeric_limits<std::size_t>::max)(), 2); });
    require_throws<std::logic_error>([] { (void)image{}.resize(1, 1); });
}

void test_crop()
{
    image original(5, 4, 4);
    for (std::size_t y = 0; y < original.height(); ++y)
        for (std::size_t x = 0; x < original.width(); ++x)
            for (std::size_t c = 0; c < original.channel(); ++c)
                original.data()[y * original.stride() + x * 4 + c] = static_cast<std::uint8_t>(y * 40 + x * 5 + c);
    const image& input = original;
    auto cropped = input.crop(1, 1, 3, 2);
    require(cropped.width() == 3 && cropped.height() == 2 && cropped.channel() == 4 && cropped.stride() == 12);
    for (std::size_t y = 0; y < cropped.height(); ++y)
        for (std::size_t x = 0; x < cropped.width(); ++x)
            for (std::size_t c = 0; c < cropped.channel(); ++c)
                require(cropped.data()[y * cropped.stride() + x * 4 + c]
                        == original.data()[(y + 1) * original.stride() + (x + 1) * 4 + c]);
    cropped.data()[0] = 0;
    require(original.data()[original.stride() + 4] == 45);
    auto full = input.crop(0, 0, 5, 4);
    require(full.data() != original.data() && full.stride() == original.stride());
    const auto edge = input.crop(4, 3, 1, 1);
    require(edge.data()[0] == 140 && edge.data()[3] == 143);
    require_throws<std::out_of_range>([&] { (void)input.crop(5, 0, 1, 1); });
    require_throws<std::out_of_range>([&] { (void)input.crop(0, 4, 1, 1); });
    require_throws<std::out_of_range>([&] { (void)input.crop(4, 3, 2, 1); });
    require_throws<std::out_of_range>([&] { (void)input.crop(4, 3, 1, 2); });
    require_throws<std::out_of_range>([&] { (void)input.crop((std::numeric_limits<std::size_t>::max)(), 0, 2, 1); });
    require_throws<std::out_of_range>([&] { (void)input.crop(1, 1, (std::numeric_limits<std::size_t>::max)(), 1); });
    require_throws<std::invalid_argument>([&] { (void)input.crop(0, 0, 0, 1); });
    require_throws<std::invalid_argument>([&] { (void)input.crop(0, 0, 1, 0); });
    require_throws<std::logic_error>([] { (void)image{}.crop(0, 0, 1, 1); });
    auto moved = std::move(original);
    require_throws<std::logic_error>([&] { (void)original.crop(0, 0, 1, 1); });
    require_throws<std::logic_error>([&] { (void)original.resize(1, 1); });
    require(moved.crop(4, 3, 1, 1).data()[0] == 140);
}

void test_image()
{
    image empty;
    require_empty(empty);
    image::shared_image shared = std::make_shared<image>(4, 3, 2);
    image::shared_const_image readonly = shared;
    image::unique_image unique = std::make_unique<image>(2, 2, 1);
    require(shared->width() == 4 && shared->height() == 3 && shared->channel() == 2);
    require(shared->stride() == 8 && readonly->data() == shared->data());
    for (std::size_t i = 0; i < shared->stride() * shared->height(); ++i)
        require(readonly->data()[i] == 0);

    shared->data()[23] = 42;
    image copied(*shared);
    require(copied.data() != shared->data() && copied.data()[23] == 42);
    copied.data()[23] = 1;
    require(readonly->data()[23] == 42);
    *unique = *shared;
    require(unique->data() != shared->data() && unique->data()[23] == 42);

    auto* address = copied.data();
    image moved(std::move(copied));
    require(moved.data() == address);
    require_empty(copied);
    empty = std::move(moved);
    require(empty.data() == address);
    require_empty(moved);
    image copied_empty(moved);
    require_empty(copied_empty);
    empty = moved;
    require_empty(empty);

    require_throws<std::invalid_argument>([] { image invalid(0, 1, 1); });
    require_throws<std::invalid_argument>([] { image invalid(1, 0, 1); });
    require_throws<std::invalid_argument>([] { image invalid(1, 1, 0); });
    require_throws<std::length_error>([] {
        image invalid((std::numeric_limits<std::size_t>::max)(), 1, 2);
    });
    require_throws<std::length_error>([] {
        image invalid((std::numeric_limits<std::size_t>::max)(), 2, 1);
    });
}
}

int main()
{
    try { test_image(); test_resize(); test_crop(); }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
