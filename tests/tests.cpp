// Minimal self-contained test suite (no external framework required).
// Run through CTest:  ctest --output-on-failure

#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "nn/datasets.hpp"
#include "nn/mnist.hpp"
#include "nn/neural_network.hpp"

namespace {

int failures = 0;

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::cerr << "    CHECK FAILED: " #cond " (line " << __LINE__ << ")\n";  \
            ++failures;                                                              \
        }                                                                            \
    } while (0)

template <typename Exception, typename Fn>
bool throwsAs(Fn&& fn) {
    try {
        fn();
    } catch (const Exception&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

void run(const std::string& name, const std::function<void()>& test) {
    const int before = failures;
    try {
        test();
    } catch (const std::exception& e) {
        std::cerr << "    unexpected exception: " << e.what() << "\n";
        ++failures;
    }
    std::cout << (failures == before ? "[ PASS ] " : "[ FAIL ] ") << name << "\n";
}

// --- Gradient checks for every supported loss / output pairing ---------------
void testGradients() {
    struct Case {
        nn::Loss loss;
        nn::OutputActivation out;
        std::vector<double> target;
    };
    const std::vector<Case> cases = {
        {nn::Loss::BinaryCrossEntropy, nn::OutputActivation::Sigmoid, {1.0}},
        {nn::Loss::MeanSquaredError, nn::OutputActivation::Sigmoid, {0.0}},
        {nn::Loss::MeanSquaredError, nn::OutputActivation::Linear, {0.37}},
    };
    for (const Case& c : cases) {
        nn::NeuralNetwork net({3, 6, 5, 1}, c.loss, 7, c.out);
        CHECK(net.gradientCheck({0.3, -0.7, 1.2}, c.target) < 1e-6);
    }
    nn::NeuralNetwork softmax({3, 6, 5, 4}, nn::Loss::CategoricalCrossEntropy, 7);
    CHECK(softmax.gradientCheck({0.3, -0.7, 1.2}, {0, 0, 1, 0}) < 1e-6);
}

// --- Softmax outputs form a probability distribution -------------------------
void testSoftmaxSumsToOne() {
    nn::NeuralNetwork net({4, 8, 5}, nn::Loss::CategoricalCrossEntropy, 3);
    const std::vector<double> out = net.predict({100.0, -100.0, 50.0, 0.0});  // extreme inputs
    double sum = 0.0;
    for (double p : out) {
        CHECK(p >= 0.0 && p <= 1.0 && std::isfinite(p));
        sum += p;
    }
    CHECK(std::abs(sum - 1.0) < 1e-12);
}

// --- Learning: XOR must be solved ---------------------------------------------
void testXorLearns() {
    const nn::Dataset data = nn::datasets::makeXor();
    nn::NeuralNetwork net({2, 8, 1}, nn::Loss::BinaryCrossEntropy, 42);
    nn::TrainConfig cfg;
    cfg.learningRate = 0.05;
    cfg.epochs = 1000;
    cfg.batchSize = 4;
    cfg.logEvery = 0;
    net.train(data, cfg);
    CHECK(net.evaluate(data).accuracy == 1.0);
}

// --- Learning: multi-class spiral with each optimizer -------------------------
void testSpiralOptimizers() {
    std::mt19937_64 rng(1);
    const nn::Dataset all = nn::datasets::makeSpiral(200, 3, 0.1, rng);
    auto [train, test] = nn::datasets::trainTestSplit(all, 0.2, rng);

    const std::vector<std::pair<nn::Optimizer, double>> optimizers = {
        {nn::Optimizer::SGD, 0.05}, {nn::Optimizer::Momentum, 0.02}, {nn::Optimizer::Adam, 0.01}};
    for (const auto& [opt, lr] : optimizers) {
        nn::NeuralNetwork net({2, 64, 32, 3}, nn::Loss::CategoricalCrossEntropy, 1);
        nn::TrainConfig cfg;
        cfg.optimizer = opt;
        cfg.learningRate = lr;
        cfg.epochs = 150;
        cfg.logEvery = 0;
        const double before = net.evaluate(test).loss;
        net.train(train, cfg);
        const nn::Metrics after = net.evaluate(test);
        CHECK(after.loss < before);
        CHECK(after.accuracy > 0.8);
    }
}

// --- Determinism: same seed -> identical results ------------------------------
void testReproducibility() {
    auto runOnce = [] {
        std::mt19937_64 rng(5);
        const nn::Dataset d = nn::datasets::makeCircle(300, rng);
        nn::NeuralNetwork net({2, 8, 1}, nn::Loss::BinaryCrossEntropy, 5);
        nn::TrainConfig cfg;
        cfg.epochs = 20;
        cfg.logEvery = 0;
        return net.train(d, cfg).trainLoss.back();
    };
    CHECK(runOnce() == runOnce());
}

// --- Persistence round-trip ---------------------------------------------------
void testSaveLoad() {
    const std::string path = "test_model_roundtrip.txt";
    nn::NeuralNetwork net({3, 7, 4}, nn::Loss::CategoricalCrossEntropy, 11);
    net.save(path);
    const nn::NeuralNetwork loaded = nn::NeuralNetwork::load(path);
    std::remove(path.c_str());

    CHECK(loaded.layerSizes() == net.layerSizes());
    CHECK(loaded.loss() == net.loss());
    CHECK(loaded.outputActivation() == net.outputActivation());
    const std::vector<double> x = {0.1, -0.4, 0.9};
    const auto a = net.predict(x);
    const auto b = loaded.predict(x);
    for (std::size_t i = 0; i < a.size(); ++i) CHECK(a[i] == b[i]);

    CHECK(throwsAs<std::runtime_error>([] { nn::NeuralNetwork::load("does_not_exist.txt"); }));
}

// --- Input validation ---------------------------------------------------------
void testValidation() {
    CHECK(throwsAs<std::invalid_argument>([] { nn::NeuralNetwork net({3}); }));
    CHECK(throwsAs<std::invalid_argument>([] { nn::NeuralNetwork net({3, 0, 1}); }));
    CHECK(throwsAs<std::invalid_argument>([] {
        nn::NeuralNetwork net({2, 4, 1}, nn::Loss::CategoricalCrossEntropy);  // softmax needs >= 2
    }));
    CHECK(throwsAs<std::invalid_argument>([] {
        nn::NeuralNetwork net({2, 4, 3}, nn::Loss::BinaryCrossEntropy, 1, nn::OutputActivation::Softmax);
    }));

    nn::NeuralNetwork net({2, 4, 1});
    CHECK(throwsAs<std::runtime_error>([&] { net.predict({1.0}); }));

    nn::Dataset bad;
    bad.inputs = {{1.0, 2.0}};
    bad.targets = {};
    CHECK(throwsAs<std::runtime_error>([&] { net.evaluate(bad); }));

    nn::TrainConfig cfg;
    cfg.batchSize = 0;
    CHECK(throwsAs<std::invalid_argument>([&] { net.train(nn::datasets::makeXor(), cfg); }));
}

// --- MNIST (IDX) loader using a tiny synthetic file ---------------------------
void writeBigEndian32(std::ofstream& f, std::uint32_t v) {
    const unsigned char b[4] = {static_cast<unsigned char>(v >> 24), static_cast<unsigned char>(v >> 16),
                                static_cast<unsigned char>(v >> 8), static_cast<unsigned char>(v)};
    f.write(reinterpret_cast<const char*>(b), 4);
}

void testMnistLoader() {
    const std::string imgPath = "test_images.idx", lblPath = "test_labels.idx";
    {
        std::ofstream img(imgPath, std::ios::binary), lbl(lblPath, std::ios::binary);
        writeBigEndian32(img, 0x803);
        writeBigEndian32(img, 2);  // 2 images of 2x2 pixels
        writeBigEndian32(img, 2);
        writeBigEndian32(img, 2);
        const unsigned char pixels[8] = {0, 255, 51, 102, 255, 0, 0, 153};
        img.write(reinterpret_cast<const char*>(pixels), 8);
        writeBigEndian32(lbl, 0x801);
        writeBigEndian32(lbl, 2);
        const unsigned char labels[2] = {3, 7};
        lbl.write(reinterpret_cast<const char*>(labels), 2);
    }
    const nn::Dataset d = nn::mnist::load(imgPath, lblPath);
    std::remove(imgPath.c_str());
    std::remove(lblPath.c_str());

    CHECK(d.size() == 2);
    CHECK(d.inputs[0].size() == 4);
    CHECK(std::abs(d.inputs[0][1] - 1.0) < 1e-12);
    CHECK(std::abs(d.inputs[0][2] - 0.2) < 1e-12);
    CHECK(d.targets[0][3] == 1.0 && d.targets[1][7] == 1.0);
    CHECK(throwsAs<std::runtime_error>([&] { nn::mnist::load("missing.idx", "missing2.idx"); }));
}

}  // namespace

int main() {
    run("gradient check: all loss/output pairings", testGradients);
    run("softmax output is a probability distribution", testSoftmaxSumsToOne);
    run("XOR is learned", testXorLearns);
    run("spiral learned by SGD, Momentum and Adam", testSpiralOptimizers);
    run("same seed gives identical training", testReproducibility);
    run("model save/load round-trip", testSaveLoad);
    run("input validation", testValidation);
    run("MNIST IDX loader", testMnistLoader);

    if (failures == 0) {
        std::cout << "\nAll tests passed.\n";
        return 0;
    }
    std::cout << "\n" << failures << " check(s) failed.\n";
    return 1;
}
