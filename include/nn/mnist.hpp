// nn/mnist.hpp
//
// Loader for the MNIST handwritten-digit dataset in its original IDX format
// (train-images-idx3-ubyte, train-labels-idx1-ubyte, ...). The files must be
// uncompressed. Pixels are scaled to [0, 1] and labels are one-hot encoded.

#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "datasets.hpp"
#include "neural_network.hpp"

namespace nn {
namespace mnist {

namespace detail {

inline std::uint32_t readBigEndian32(std::ifstream& f) {
    unsigned char b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    if (!f) throw std::runtime_error("Unexpected end of IDX file");
    return (static_cast<std::uint32_t>(b[0]) << 24) | (static_cast<std::uint32_t>(b[1]) << 16) |
           (static_cast<std::uint32_t>(b[2]) << 8) | static_cast<std::uint32_t>(b[3]);
}

}  // namespace detail

// Loads up to `limit` samples (0 = all). Inputs have 784 features.
inline Dataset load(const std::string& imagesPath, const std::string& labelsPath,
                    std::size_t limit = 0) {
    std::ifstream img(imagesPath, std::ios::binary);
    if (!img) throw std::runtime_error("Cannot open images file: " + imagesPath);
    std::ifstream lbl(labelsPath, std::ios::binary);
    if (!lbl) throw std::runtime_error("Cannot open labels file: " + labelsPath);

    if (detail::readBigEndian32(img) != 0x00000803u) {
        throw std::runtime_error("Bad magic number in images file (is it gunzipped?): " + imagesPath);
    }
    const std::size_t imgCount = detail::readBigEndian32(img);
    const std::size_t rows = detail::readBigEndian32(img);
    const std::size_t cols = detail::readBigEndian32(img);

    if (detail::readBigEndian32(lbl) != 0x00000801u) {
        throw std::runtime_error("Bad magic number in labels file (is it gunzipped?): " + labelsPath);
    }
    const std::size_t lblCount = detail::readBigEndian32(lbl);
    if (imgCount != lblCount) throw std::runtime_error("Image and label counts differ");

    const std::size_t n = (limit == 0 || limit > imgCount) ? imgCount : limit;
    const std::size_t pixels = rows * cols;

    Dataset d;
    d.inputs.reserve(n);
    d.targets.reserve(n);
    std::vector<unsigned char> buffer(pixels);
    for (std::size_t i = 0; i < n; ++i) {
        img.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(pixels));
        unsigned char label = 0;
        lbl.read(reinterpret_cast<char*>(&label), 1);
        if (!img || !lbl) throw std::runtime_error("Truncated IDX file");

        std::vector<double> x(pixels);
        for (std::size_t p = 0; p < pixels; ++p) x[p] = static_cast<double>(buffer[p]) / 255.0;
        d.inputs.push_back(std::move(x));
        d.targets.push_back(datasets::oneHot(label, 10));
    }
    return d;
}

}  // namespace mnist
}  // namespace nn
