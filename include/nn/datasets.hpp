// nn/datasets.hpp
//
// Small helpers for building and splitting datasets: synthetic benchmarks
// (circle, XOR, spiral) and utilities (one-hot encoding, train/test split).

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "neural_network.hpp"

namespace nn {
namespace datasets {

inline std::vector<double> oneHot(std::size_t label, std::size_t classes) {
    if (label >= classes) throw std::invalid_argument("oneHot: label out of range");
    std::vector<double> v(classes, 0.0);
    v[label] = 1.0;
    return v;
}

// Binary task: is the point (x, y) inside the unit circle? Points are drawn
// uniformly from [-2, 2]^2. A classic non-linearly-separable problem.
inline Dataset makeCircle(std::size_t n, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    Dataset d;
    for (std::size_t i = 0; i < n; ++i) {
        const double x = dist(rng), y = dist(rng);
        d.inputs.push_back({x, y});
        d.targets.push_back({std::sqrt(x * x + y * y) < 1.0 ? 1.0 : 0.0});
    }
    return d;
}

// The four-sample XOR truth table.
inline Dataset makeXor() {
    Dataset d;
    d.inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    d.targets = {{0}, {1}, {1}, {0}};
    return d;
}

// Multi-class spiral: `classes` interleaved arms with `perClass` points each and
// Gaussian noise. Targets are one-hot vectors.
inline Dataset makeSpiral(std::size_t perClass, std::size_t classes, double noise,
                          std::mt19937_64& rng) {
    std::normal_distribution<double> gauss(0.0, noise);
    Dataset d;
    for (std::size_t k = 0; k < classes; ++k) {
        for (std::size_t i = 0; i < perClass; ++i) {
            const double frac = static_cast<double>(i) / static_cast<double>(perClass - 1);
            const double r = frac;
            const double t = (static_cast<double>(k) + frac) * 4.0 + gauss(rng);
            d.inputs.push_back({r * std::sin(t), r * std::cos(t)});
            d.targets.push_back(oneHot(k, classes));
        }
    }
    return d;
}

// Shuffles and splits a dataset. `testRatio` is the fraction that goes to the test set.
inline std::pair<Dataset, Dataset> trainTestSplit(const Dataset& all, double testRatio,
                                                  std::mt19937_64& rng) {
    if (testRatio < 0.0 || testRatio > 1.0) {
        throw std::invalid_argument("trainTestSplit: testRatio must be in [0, 1]");
    }
    std::vector<std::size_t> idx(all.size());
    std::iota(idx.begin(), idx.end(), std::size_t{0});
    std::shuffle(idx.begin(), idx.end(), rng);

    const std::size_t testCount =
        static_cast<std::size_t>(testRatio * static_cast<double>(all.size()));
    Dataset train, test;
    for (std::size_t k = 0; k < idx.size(); ++k) {
        Dataset& dst = (k < testCount) ? test : train;
        dst.inputs.push_back(all.inputs[idx[k]]);
        dst.targets.push_back(all.targets[idx[k]]);
    }
    return {std::move(train), std::move(test)};
}

}  // namespace datasets
}  // namespace nn
