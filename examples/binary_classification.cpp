// Binary classification with Sigmoid + Binary Cross-Entropy.
//   1. Gradient check (backprop vs finite differences)
//   2. XOR
//   3. Points inside / outside the unit circle (80/20 train/test split)
//
// Usage: ./binary_classification [seed]

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>

#include "nn/datasets.hpp"
#include "nn/neural_network.hpp"

namespace {

void demoGradientCheck(std::uint64_t seed) {
    std::cout << "=== Gradient check ===\n";
    nn::NeuralNetwork net({3, 5, 4, 1}, nn::Loss::BinaryCrossEntropy, seed);
    const double err = net.gradientCheck({0.3, -0.7, 1.2}, {1.0});
    std::cout << "max relative error: " << std::scientific << std::setprecision(2) << err
              << (err < 1e-6 ? "  [OK]" : "  [FAIL]") << "\n\n";
}

void demoXor(std::uint64_t seed) {
    std::cout << "=== XOR (4 samples; train == test by nature of the problem) ===\n";
    const nn::Dataset data = nn::datasets::makeXor();

    nn::NeuralNetwork net({2, 8, 1}, nn::Loss::BinaryCrossEntropy, seed);
    nn::TrainConfig cfg;
    cfg.learningRate = 0.05;
    cfg.epochs = 1000;
    cfg.batchSize = 4;
    cfg.logEvery = 500;
    net.train(data, cfg);

    for (std::size_t k = 0; k < data.size(); ++k) {
        std::cout << "  (" << data.inputs[k][0] << ", " << data.inputs[k][1] << ") -> "
                  << std::fixed << std::setprecision(4) << net.predict(data.inputs[k])[0]
                  << "  (expected " << data.targets[k][0] << ")\n";
    }
    std::cout << "  accuracy: " << std::setprecision(1) << net.evaluate(data).accuracy * 100.0
              << "%\n\n";
}

void demoCircle(std::uint64_t seed) {
    std::cout << "=== Circle classification (80/20 train/test split) ===\n";
    std::mt19937_64 rng(seed);
    const nn::Dataset all = nn::datasets::makeCircle(2000, rng);
    auto [train, test] = nn::datasets::trainTestSplit(all, 0.2, rng);
    std::cout << "train: " << train.size() << " samples, test: " << test.size() << " samples\n";

    nn::NeuralNetwork net({2, 16, 8, 1}, nn::Loss::BinaryCrossEntropy, seed);
    nn::TrainConfig cfg;
    cfg.learningRate = 0.01;
    cfg.epochs = 300;
    cfg.batchSize = 32;
    cfg.optimizer = nn::Optimizer::Adam;
    cfg.logEvery = 50;

    const auto start = std::chrono::steady_clock::now();
    net.train(train, cfg, &test);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start)
                        .count();

    std::cout << std::fixed << std::setprecision(2) << "\ntraining time : " << ms << " ms\n"
              << "train accuracy: " << net.evaluate(train).accuracy * 100.0 << "%\n"
              << "test accuracy : " << net.evaluate(test).accuracy * 100.0 << "%\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::uint64_t seed = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 42;
        std::cout << "seed = " << seed << "\n\n";
        demoGradientCheck(seed);
        demoXor(seed);
        demoCircle(seed);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
