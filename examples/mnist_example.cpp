// MNIST handwritten digits with a {784, 128, 64, 10} network (Softmax + CCE).
//
// Download the four IDX files, gunzip them into one directory, then:
//   ./mnist_example <data_dir> [epochs=5] [train_limit=0 (all)] [seed=42]
//
// The trained model is saved to mnist_model.txt and reloaded to verify that
// persistence round-trips exactly.

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "nn/mnist.hpp"
#include "nn/neural_network.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <mnist_dir> [epochs] [train_limit] [seed]\n"
                  << "Expected files in <mnist_dir> (gunzipped):\n"
                  << "  train-images-idx3-ubyte  train-labels-idx1-ubyte\n"
                  << "  t10k-images-idx3-ubyte   t10k-labels-idx1-ubyte\n";
        return 1;
    }
    try {
        const std::string dir = argv[1];
        const std::size_t epochs = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : 5;
        const std::size_t limit = (argc > 3) ? std::strtoull(argv[3], nullptr, 10) : 0;
        const std::uint64_t seed = (argc > 4) ? std::strtoull(argv[4], nullptr, 10) : 42;

        std::cout << "Loading MNIST from " << dir << " ...\n";
        const nn::Dataset train =
            nn::mnist::load(dir + "/train-images-idx3-ubyte", dir + "/train-labels-idx1-ubyte", limit);
        const nn::Dataset test =
            nn::mnist::load(dir + "/t10k-images-idx3-ubyte", dir + "/t10k-labels-idx1-ubyte");
        std::cout << "train: " << train.size() << " samples, test: " << test.size() << " samples\n";

        nn::NeuralNetwork net({784, 128, 64, 10}, nn::Loss::CategoricalCrossEntropy, seed);
        std::cout << "parameters: " << net.parameterCount() << "\n\n";

        nn::TrainConfig cfg;
        cfg.optimizer = nn::Optimizer::Adam;
        cfg.learningRate = 0.001;
        cfg.epochs = epochs;
        cfg.batchSize = 64;
        cfg.logEvery = 1;

        const auto start = std::chrono::steady_clock::now();
        net.train(train, cfg, &test);
        const auto sec = std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::steady_clock::now() - start)
                             .count();

        const nn::Metrics m = net.evaluate(test);
        std::cout << std::fixed << std::setprecision(2) << "\ntraining time: " << sec << " s\n"
                  << "test accuracy: " << m.accuracy * 100.0 << "%\n";

        net.save("mnist_model.txt");
        const nn::NeuralNetwork reloaded = nn::NeuralNetwork::load("mnist_model.txt");
        const nn::Metrics m2 = reloaded.evaluate(test);
        std::cout << "reloaded model test accuracy: " << m2.accuracy * 100.0 << "%"
                  << (m2.accuracy == m.accuracy ? "  [identical]" : "  [MISMATCH]") << "\n";
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
