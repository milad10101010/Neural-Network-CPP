// Multi-class classification (3-arm spiral) with Softmax + Categorical
// Cross-Entropy, comparing the three optimizers on identical data and seed.
//
// Usage: ./spiral_multiclass [seed]

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include "nn/datasets.hpp"
#include "nn/neural_network.hpp"

int main(int argc, char** argv) {
    try {
        const std::uint64_t seed = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 42;
        std::mt19937_64 rng(seed);

        const nn::Dataset all = nn::datasets::makeSpiral(/*perClass=*/300, /*classes=*/3, /*noise=*/0.2, rng);
        auto [train, test] = nn::datasets::trainTestSplit(all, 0.2, rng);
        std::cout << "seed = " << seed << " | train: " << train.size() << ", test: " << test.size()
                  << " samples, 3 classes\n\n";

        struct Run {
            const char* name;
            nn::Optimizer optimizer;
            double lr;
        };
        const Run runs[] = {{"SGD", nn::Optimizer::SGD, 0.05},
                            {"Momentum", nn::Optimizer::Momentum, 0.02},
                            {"Adam", nn::Optimizer::Adam, 0.01}};

        std::cout << std::left << std::setw(10) << "optimizer" << std::right << std::setw(14)
                  << "train acc" << std::setw(14) << "test acc" << std::setw(14) << "test loss\n";

        for (const Run& run : runs) {
            nn::NeuralNetwork net({2, 64, 32, 3}, nn::Loss::CategoricalCrossEntropy, seed);
            nn::TrainConfig cfg;
            cfg.optimizer = run.optimizer;
            cfg.learningRate = run.lr;
            cfg.epochs = 200;
            cfg.batchSize = 32;
            cfg.logEvery = 0;  // quiet
            net.train(train, cfg);

            const nn::Metrics tr = net.evaluate(train);
            const nn::Metrics te = net.evaluate(test);
            std::cout << std::left << std::setw(10) << run.name << std::right << std::fixed
                      << std::setprecision(2) << std::setw(13) << tr.accuracy * 100.0 << "%"
                      << std::setw(13) << te.accuracy * 100.0 << "%" << std::setprecision(4)
                      << std::setw(14) << te.loss << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
