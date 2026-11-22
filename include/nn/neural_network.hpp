// nn/neural_network.hpp
//
// A dependency-free, header-only feedforward neural network written from
// scratch in C++17. Designed to be read: every step of forward propagation,
// backpropagation and optimisation is explicit.
//
// Highlights
//   * Arbitrary depth (any list of layer sizes)
//   * Hidden layers: ReLU.  Output layers: Sigmoid, Softmax or Linear
//   * Losses: Binary Cross-Entropy, Categorical Cross-Entropy, MSE
//   * Optimizers: SGD, SGD + Momentum, Adam (with optional L2 weight decay)
//   * Mini-batch training with per-epoch shuffling
//   * Deterministic runs through an explicit seed
//   * Numerical gradient check, model save / load
//
// License: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nn {

// ---------------------------------------------------------------------------
// Scalar activation functions
// ---------------------------------------------------------------------------
namespace activation {

inline double relu(double x) { return x > 0.0 ? x : 0.0; }
inline double reluDerivative(double z) { return z > 0.0 ? 1.0 : 0.0; }

// Numerically stable sigmoid: never calls exp() with a large positive argument.
inline double sigmoid(double x) {
    if (x >= 0.0) {
        const double e = std::exp(-x);
        return 1.0 / (1.0 + e);
    }
    const double e = std::exp(x);
    return e / (1.0 + e);
}

}  // namespace activation

// ---------------------------------------------------------------------------
// Contiguous row-major matrix
// ---------------------------------------------------------------------------
class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), data_(rows * cols, 0.0) {}

    double& operator()(std::size_t i, std::size_t j) { return data_[i * cols_ + j]; }
    const double& operator()(std::size_t i, std::size_t j) const { return data_[i * cols_ + j]; }

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }

    // Flat access, so optimizers can treat weights and biases uniformly.
    std::vector<double>& raw() { return data_; }
    const std::vector<double>& raw() const { return data_; }

private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<double> data_;
};

// ---------------------------------------------------------------------------
// Public configuration types
// ---------------------------------------------------------------------------
enum class Loss { MeanSquaredError = 0, BinaryCrossEntropy = 1, CategoricalCrossEntropy = 2 };

// Auto picks the natural pairing: BCE -> Sigmoid, CCE -> Softmax, MSE -> Linear.
enum class OutputActivation { Auto = 0, Sigmoid = 1, Softmax = 2, Linear = 3 };

enum class Optimizer { SGD, Momentum, Adam };

struct Dataset {
    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> targets;  // one-hot for classification
    std::size_t size() const { return inputs.size(); }
};

struct TrainConfig {
    double learningRate = 0.01;
    std::size_t epochs = 100;
    std::size_t batchSize = 32;
    Optimizer optimizer = Optimizer::Adam;
    double momentum = 0.9;     // SGD + Momentum
    double beta1 = 0.9;        // Adam
    double beta2 = 0.999;      // Adam
    double epsilon = 1e-8;     // Adam
    double weightDecay = 0.0;  // L2 regularisation strength (weights only, not biases)
    bool shuffle = true;
    std::size_t logEvery = 50;  // 0 disables logging
};

struct Metrics {
    double loss = 0.0;
    // Fraction of correct predictions. NaN for Linear outputs (regression).
    double accuracy = std::numeric_limits<double>::quiet_NaN();
};

struct History {
    std::vector<double> trainLoss;
    std::vector<double> valLoss;  // empty when no validation set is given
};

// ---------------------------------------------------------------------------
// Neural network
// ---------------------------------------------------------------------------
class NeuralNetwork {
public:
    NeuralNetwork(std::vector<std::size_t> layerSizes,
                  Loss loss = Loss::BinaryCrossEntropy,
                  std::uint64_t seed = 42,
                  OutputActivation output = OutputActivation::Auto)
        : sizes_(std::move(layerSizes)), loss_(loss), out_(resolveOutput(loss, output)), rng_(seed) {
        if (sizes_.size() < 2) {
            throw std::invalid_argument("Network needs at least an input and an output layer");
        }
        for (std::size_t s : sizes_) {
            if (s == 0) throw std::invalid_argument("Layer sizes must be positive");
        }
        validatePairing();

        layers_.resize(sizes_.size() - 1);
        for (std::size_t l = 0; l < layers_.size(); ++l) {
            const std::size_t in = sizes_[l];
            const std::size_t out = sizes_[l + 1];
            Layer& layer = layers_[l];

            layer.W = Matrix(out, in);
            layer.gW = Matrix(out, in);
            layer.mW = Matrix(out, in);
            layer.vW = Matrix(out, in);
            layer.b.assign(out, 0.0);
            layer.gb.assign(out, 0.0);
            layer.mb.assign(out, 0.0);
            layer.vb.assign(out, 0.0);

            // He initialisation for ReLU layers, Xavier/Glorot for the output layer.
            const bool isOutput = (l + 1 == layers_.size());
            const double stddev = isOutput ? std::sqrt(2.0 / static_cast<double>(in + out))
                                           : std::sqrt(2.0 / static_cast<double>(in));
            std::normal_distribution<double> dist(0.0, stddev);
            for (double& w : layer.W.raw()) w = dist(rng_);
        }
    }

    // ----- Introspection -----------------------------------------------------
    const std::vector<std::size_t>& layerSizes() const { return sizes_; }
    Loss loss() const { return loss_; }
    OutputActivation outputActivation() const { return out_; }

    std::size_t parameterCount() const {
        std::size_t n = 0;
        for (const Layer& layer : layers_) n += layer.W.raw().size() + layer.b.size();
        return n;
    }

    // ----- Inference ---------------------------------------------------------
    std::vector<double> predict(const std::vector<double>& input) const {
        Cache cache;
        forwardPass(input, cache);
        return cache.a.back();
    }

    // Index of the most probable class (argmax of the output vector).
    std::size_t predictClass(const std::vector<double>& input) const {
        const std::vector<double> out = predict(input);
        return argmax(out);
    }

    // ----- Evaluation --------------------------------------------------------
    Metrics evaluate(const Dataset& data) const {
        validateDataset(data);
        Metrics m;
        if (data.size() == 0) return m;
        Cache cache;
        std::size_t correct = 0;
        for (std::size_t k = 0; k < data.size(); ++k) {
            forwardPass(data.inputs[k], cache);
            const auto& out = cache.a.back();
            m.loss += sampleLoss(out, data.targets[k]);
            if (isCorrect(out, data.targets[k])) ++correct;
        }
        m.loss /= static_cast<double>(data.size());
        if (out_ != OutputActivation::Linear) {
            m.accuracy = static_cast<double>(correct) / static_cast<double>(data.size());
        }
        return m;
    }

    // ----- Training ----------------------------------------------------------
    History train(const Dataset& trainSet, const TrainConfig& cfg, const Dataset* valSet = nullptr) {
        validateDataset(trainSet);
        if (valSet) validateDataset(*valSet);
        if (trainSet.size() == 0) throw std::invalid_argument("Training set is empty");
        if (cfg.epochs == 0 || cfg.batchSize == 0 || cfg.learningRate <= 0.0) {
            throw std::invalid_argument("epochs, batchSize and learningRate must be positive");
        }

        History history;
        std::vector<std::size_t> order(trainSet.size());
        std::iota(order.begin(), order.end(), std::size_t{0});
        Cache cache;

        for (std::size_t epoch = 0; epoch < cfg.epochs; ++epoch) {
            if (cfg.shuffle) std::shuffle(order.begin(), order.end(), rng_);

            double epochLoss = 0.0;
            for (std::size_t start = 0; start < order.size(); start += cfg.batchSize) {
                const std::size_t end = std::min(order.size(), start + cfg.batchSize);
                for (std::size_t k = start; k < end; ++k) {
                    const std::size_t idx = order[k];
                    forwardPass(trainSet.inputs[idx], cache);
                    epochLoss += sampleLoss(cache.a.back(), trainSet.targets[idx]);
                    backward(trainSet.targets[idx], cache);
                }
                applyUpdate(cfg, end - start);
            }
            epochLoss /= static_cast<double>(trainSet.size());
            history.trainLoss.push_back(epochLoss);

            Metrics val;
            if (valSet) {
                val = evaluate(*valSet);
                history.valLoss.push_back(val.loss);
            }

            if (cfg.logEvery != 0 && (epoch % cfg.logEvery == 0 || epoch + 1 == cfg.epochs)) {
                std::cout << "epoch " << std::setw(4) << epoch + 1 << "/" << cfg.epochs
                          << " | train loss " << std::fixed << std::setprecision(5) << epochLoss;
                if (valSet) {
                    std::cout << " | val loss " << val.loss;
                    if (!std::isnan(val.accuracy)) {
                        std::cout << " | val acc " << std::setprecision(2) << val.accuracy * 100.0 << "%";
                    }
                }
                std::cout << std::endl;
            }
        }
        return history;
    }

    // ----- Gradient check ----------------------------------------------------
    // Compares backpropagation gradients with central finite differences and
    // returns the maximum relative error (healthy values are below ~1e-6).
    double gradientCheck(const std::vector<double>& input, const std::vector<double>& target,
                         double h = 1e-5) {
        zeroGradients();
        Cache cache;
        forwardPass(input, cache);
        backward(target, cache);

        double maxRelError = 0.0;
        auto check = [&](std::vector<double>& params, const std::vector<double>& grads) {
            for (std::size_t k = 0; k < params.size(); ++k) {
                const double original = params[k];
                params[k] = original + h;
                const double lossPlus = lossOf(input, target);
                params[k] = original - h;
                const double lossMinus = lossOf(input, target);
                params[k] = original;

                const double numeric = (lossPlus - lossMinus) / (2.0 * h);
                const double denom = std::max(1e-8, std::abs(numeric) + std::abs(grads[k]));
                maxRelError = std::max(maxRelError, std::abs(numeric - grads[k]) / denom);
            }
        };
        for (Layer& layer : layers_) {
            check(layer.W.raw(), layer.gW.raw());
            check(layer.b, layer.gb);
        }
        zeroGradients();
        return maxRelError;
    }

    // ----- Persistence (plain text, full double precision) --------------------
    void save(const std::string& path) const {
        std::ofstream f(path);
        if (!f) throw std::runtime_error("Cannot open file for writing: " + path);
        f << "NNCPP 1\n" << static_cast<int>(loss_) << ' ' << static_cast<int>(out_) << '\n';
        f << sizes_.size();
        for (std::size_t s : sizes_) f << ' ' << s;
        f << '\n' << std::setprecision(17);
        for (const Layer& layer : layers_) {
            for (double w : layer.W.raw()) f << w << ' ';
            f << '\n';
            for (double b : layer.b) f << b << ' ';
            f << '\n';
        }
        if (!f) throw std::runtime_error("Failed while writing: " + path);
    }

    static NeuralNetwork load(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("Cannot open model file: " + path);

        std::string magic;
        int version = 0, loss = 0, out = 0;
        std::size_t count = 0;
        f >> magic >> version >> loss >> out >> count;
        if (!f || magic != "NNCPP" || version != 1 || count < 2) {
            throw std::runtime_error("Not a valid model file: " + path);
        }
        std::vector<std::size_t> sizes(count);
        for (std::size_t& s : sizes) f >> s;
        if (!f) throw std::runtime_error("Corrupt model header: " + path);

        NeuralNetwork net(sizes, static_cast<Loss>(loss), 0, static_cast<OutputActivation>(out));
        for (Layer& layer : net.layers_) {
            for (double& w : layer.W.raw()) f >> w;
            for (double& b : layer.b) f >> b;
        }
        if (!f) throw std::runtime_error("Corrupt or truncated model file: " + path);
        return net;
    }

private:
    struct Layer {
        Matrix W, gW, mW, vW;  // weights, gradient accumulator, optimizer moments
        std::vector<double> b, gb, mb, vb;
    };

    // z[l]    = pre-activation of layer l
    // a[0]    = network input, a[l + 1] = output of layer l
    struct Cache {
        std::vector<std::vector<double>> z;
        std::vector<std::vector<double>> a;
    };

    std::vector<std::size_t> sizes_;
    Loss loss_;
    OutputActivation out_;
    std::mt19937_64 rng_;
    std::vector<Layer> layers_;
    std::size_t step_ = 0;  // Adam time step

    // ----- Configuration helpers ---------------------------------------------
    static OutputActivation resolveOutput(Loss loss, OutputActivation requested) {
        if (requested != OutputActivation::Auto) return requested;
        switch (loss) {
            case Loss::BinaryCrossEntropy: return OutputActivation::Sigmoid;
            case Loss::CategoricalCrossEntropy: return OutputActivation::Softmax;
            case Loss::MeanSquaredError: return OutputActivation::Linear;
        }
        return OutputActivation::Linear;
    }

    void validatePairing() const {
        const bool ok =
            (loss_ == Loss::BinaryCrossEntropy && out_ == OutputActivation::Sigmoid) ||
            (loss_ == Loss::CategoricalCrossEntropy && out_ == OutputActivation::Softmax) ||
            (loss_ == Loss::MeanSquaredError &&
             (out_ == OutputActivation::Sigmoid || out_ == OutputActivation::Linear));
        if (!ok) {
            throw std::invalid_argument(
                "Unsupported loss/output pairing. Use BCE+Sigmoid, CCE+Softmax, MSE+Sigmoid or MSE+Linear");
        }
        if (out_ == OutputActivation::Softmax && sizes_.back() < 2) {
            throw std::invalid_argument("Softmax needs at least 2 output neurons");
        }
    }

    // ----- Forward pass (the ONE implementation) -------------------------------
    void forwardPass(const std::vector<double>& input, Cache& cache) const {
        if (input.size() != sizes_.front()) {
            throw std::runtime_error("Input size mismatch");
        }
        const std::size_t L = layers_.size();
        cache.z.resize(L);
        cache.a.resize(L + 1);
        cache.a[0] = input;

        for (std::size_t l = 0; l < L; ++l) {
            const Layer& layer = layers_[l];
            const std::size_t out = layer.W.rows();
            const std::size_t in = layer.W.cols();
            const bool isOutput = (l + 1 == L);

            std::vector<double>& z = cache.z[l];
            std::vector<double>& a = cache.a[l + 1];
            z.assign(out, 0.0);
            a.assign(out, 0.0);

            for (std::size_t j = 0; j < out; ++j) {
                double sum = layer.b[j];
                for (std::size_t i = 0; i < in; ++i) sum += layer.W(j, i) * cache.a[l][i];
                z[j] = sum;
            }

            if (!isOutput) {
                for (std::size_t j = 0; j < out; ++j) a[j] = activation::relu(z[j]);
            } else {
                applyOutputActivation(z, a);
            }
        }
    }

    void applyOutputActivation(const std::vector<double>& z, std::vector<double>& a) const {
        switch (out_) {
            case OutputActivation::Sigmoid:
                for (std::size_t j = 0; j < z.size(); ++j) a[j] = activation::sigmoid(z[j]);
                break;
            case OutputActivation::Linear:
                a = z;
                break;
            case OutputActivation::Softmax: {
                const double maxZ = *std::max_element(z.begin(), z.end());  // stability
                double total = 0.0;
                for (std::size_t j = 0; j < z.size(); ++j) {
                    a[j] = std::exp(z[j] - maxZ);
                    total += a[j];
                }
                for (double& v : a) v /= total;
                break;
            }
            case OutputActivation::Auto:
                break;  // resolved in the constructor, never reached
        }
    }

    // ----- Backpropagation: accumulates one sample's gradients ----------------
    void backward(const std::vector<double>& target, const Cache& cache) {
        const std::size_t L = layers_.size();
        const auto& output = cache.a[L];

        // dLoss/dz at the output layer.
        // For (Sigmoid+BCE), (Softmax+CCE) and (Linear+MSE) it simplifies to
        // (output - target). Only Sigmoid+MSE needs the extra sigmoid derivative.
        std::vector<double> delta(output.size());
        for (std::size_t j = 0; j < output.size(); ++j) {
            const double diff = output[j] - target[j];
            const bool sigmoidMse =
                (out_ == OutputActivation::Sigmoid && loss_ == Loss::MeanSquaredError);
            delta[j] = sigmoidMse ? diff * output[j] * (1.0 - output[j]) : diff;
        }

        for (std::size_t l = L; l-- > 0;) {
            Layer& layer = layers_[l];
            const auto& prevA = cache.a[l];
            const std::size_t out = layer.W.rows();
            const std::size_t in = layer.W.cols();

            for (std::size_t j = 0; j < out; ++j) {
                layer.gb[j] += delta[j];
                for (std::size_t i = 0; i < in; ++i) layer.gW(j, i) += delta[j] * prevA[i];
            }

            if (l > 0) {
                std::vector<double> prevDelta(in, 0.0);
                for (std::size_t j = 0; j < out; ++j) {
                    for (std::size_t i = 0; i < in; ++i) prevDelta[i] += layer.W(j, i) * delta[j];
                }
                for (std::size_t i = 0; i < in; ++i) {
                    prevDelta[i] *= activation::reluDerivative(cache.z[l - 1][i]);
                }
                delta = std::move(prevDelta);
            }
        }
    }

    // ----- Optimizer step -------------------------------------------------------
    void applyUpdate(const TrainConfig& cfg, std::size_t batchCount) {
        ++step_;
        const double invBatch = 1.0 / static_cast<double>(batchCount);
        const double correction1 = 1.0 - std::pow(cfg.beta1, static_cast<double>(step_));
        const double correction2 = 1.0 - std::pow(cfg.beta2, static_cast<double>(step_));

        auto update = [&](std::vector<double>& p, std::vector<double>& g, std::vector<double>& m,
                          std::vector<double>& v, bool decay) {
            for (std::size_t k = 0; k < p.size(); ++k) {
                double grad = g[k] * invBatch;
                if (decay) grad += cfg.weightDecay * p[k];
                switch (cfg.optimizer) {
                    case Optimizer::SGD:
                        p[k] -= cfg.learningRate * grad;
                        break;
                    case Optimizer::Momentum:
                        m[k] = cfg.momentum * m[k] + grad;
                        p[k] -= cfg.learningRate * m[k];
                        break;
                    case Optimizer::Adam: {
                        m[k] = cfg.beta1 * m[k] + (1.0 - cfg.beta1) * grad;
                        v[k] = cfg.beta2 * v[k] + (1.0 - cfg.beta2) * grad * grad;
                        const double mHat = m[k] / correction1;
                        const double vHat = v[k] / correction2;
                        p[k] -= cfg.learningRate * mHat / (std::sqrt(vHat) + cfg.epsilon);
                        break;
                    }
                }
                g[k] = 0.0;  // reset the accumulator for the next batch
            }
        };

        for (Layer& layer : layers_) {
            update(layer.W.raw(), layer.gW.raw(), layer.mW.raw(), layer.vW.raw(), true);
            update(layer.b, layer.gb, layer.mb, layer.vb, false);
        }
    }

    void zeroGradients() {
        for (Layer& layer : layers_) {
            std::fill(layer.gW.raw().begin(), layer.gW.raw().end(), 0.0);
            std::fill(layer.gb.begin(), layer.gb.end(), 0.0);
        }
    }

    // ----- Loss and metrics -----------------------------------------------------
    double sampleLoss(const std::vector<double>& out, const std::vector<double>& target) const {
        constexpr double kEps = 1e-12;
        double sum = 0.0;
        for (std::size_t j = 0; j < out.size(); ++j) {
            switch (loss_) {
                case Loss::BinaryCrossEntropy: {
                    const double o = std::min(std::max(out[j], kEps), 1.0 - kEps);
                    sum -= target[j] * std::log(o) + (1.0 - target[j]) * std::log(1.0 - o);
                    break;
                }
                case Loss::CategoricalCrossEntropy: {
                    const double o = std::max(out[j], kEps);
                    sum -= target[j] * std::log(o);
                    break;
                }
                case Loss::MeanSquaredError: {
                    const double d = out[j] - target[j];
                    sum += 0.5 * d * d;
                    break;
                }
            }
        }
        return sum;
    }

    double lossOf(const std::vector<double>& input, const std::vector<double>& target) const {
        Cache cache;
        forwardPass(input, cache);
        return sampleLoss(cache.a.back(), target);
    }

    static std::size_t argmax(const std::vector<double>& v) {
        return static_cast<std::size_t>(std::max_element(v.begin(), v.end()) - v.begin());
    }

    // Single output: threshold at 0.5. Multiple outputs: compare argmax.
    static bool isCorrect(const std::vector<double>& out, const std::vector<double>& target) {
        if (out.size() == 1) return (out[0] >= 0.5) == (target[0] >= 0.5);
        return argmax(out) == argmax(target);
    }

    void validateDataset(const Dataset& data) const {
        if (data.inputs.size() != data.targets.size()) {
            throw std::runtime_error("Inputs and targets must have the same number of samples");
        }
        for (std::size_t k = 0; k < data.size(); ++k) {
            if (data.inputs[k].size() != sizes_.front()) {
                throw std::runtime_error("Sample " + std::to_string(k) + ": input size mismatch");
            }
            if (data.targets[k].size() != sizes_.back()) {
                throw std::runtime_error("Sample " + std::to_string(k) + ": target size mismatch");
            }
        }
    }
};

}  // namespace nn
