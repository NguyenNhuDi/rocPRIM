// MIT License
//
// Copyright (c) 2017-2019 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <locale>
#include <codecvt>
#include <string>

// Google Benchmark
#include "benchmark/benchmark.h"

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM HIP API
#include <rocprim/rocprim.hpp>

// CmdParser
#include "cmdparser.hpp"
#include "benchmark_utils.hpp"
#include "benchmark_device_reduce.parallel.hpp"

#define HIP_CHECK(condition)         \
  {                                  \
    hipError_t error = condition;    \
    if(error != hipSuccess){         \
        std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
        exit(error); \
    } \
  }

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 128;
#endif

#define CREATE_BENCHMARK(T, REDUCE_OP)                                                          \
{                                                                                               \
    const device_reduce_benchmark<T, REDUCE_OP> instance;                                       \
    benchmark::internal::Benchmark* benchmark = benchmark::RegisterBenchmark(                   \
        instance.name().c_str(),                                                                \
        [instance](benchmark::State& state, size_t size, const hipStream_t stream) {            \
            instance.run(state, size, stream);                                                  \
        },                                                                                      \
        size,                                                                                   \
        stream);                                                                                \
    benchmarks.emplace_back(benchmark);                                                         \
}

int main(int argc, char *argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_N, "number of values");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.set_optional<int>("parallel_instance", "parallel_instance", 0, "parallel instance");
    parser.set_optional<int>("parallel_instances", "parallel_instances", 1, "total parallel instances");
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t size = parser.get<size_t>("size");
    const int trials = parser.get<int>("trials");

    // HIP
    hipStream_t stream = 0; // default
    hipDeviceProp_t devProp;
    int device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));
    std::cout << "[HIP] Device name: " << devProp.name << std::endl;

    std::vector<benchmark::internal::Benchmark*> benchmarks = {};
#ifdef BENCHMARK_CONFIG_TUNING
    const int parallel_instance = parser.get<int>("parallel_instance");
    const int parallel_instances = parser.get<int>("parallel_instances");
    std::vector<std::unique_ptr<config_autotune_interface>>& configs = config_autotune_register::vector();
    // sorting to get a consistent order because order of initialization of static variables is undefined by the C++ standard.
    std::sort(configs.begin(), configs.end(), [](const auto& l, const auto& r){
        return l->name() < r->name();
    });
    size_t configs_per_instance = (configs.size() + parallel_instances - 1) / parallel_instances;
    size_t start = std::min(parallel_instance * configs_per_instance, configs.size());
    size_t end   = std::min((parallel_instance + 1) * configs_per_instance, configs.size());

    for(size_t i = start; i < end; i++)
    {
        std::unique_ptr<config_autotune_interface>& uniq_ptr = configs.at(i);
        config_autotune_interface* tuning_benchmark = uniq_ptr.get();
        benchmark::internal::Benchmark* benchmark = benchmark::RegisterBenchmark(
            tuning_benchmark->name().c_str(),
            [tuning_benchmark](benchmark::State& state, size_t size, const hipStream_t stream) {
                tuning_benchmark->run(state, size, stream);
            },
            size,
            stream);
        benchmarks.emplace_back(benchmark);
    }
#else
    using custom_float2 = custom_type<float, float>;
    using custom_double2 = custom_type<double, double>;
    // Add benchmarks
    CREATE_BENCHMARK(int, rocprim::plus<int>)
    CREATE_BENCHMARK(long long, rocprim::plus<long long>)

    CREATE_BENCHMARK(float, rocprim::plus<float>)
    CREATE_BENCHMARK(double, rocprim::plus<double>)

    CREATE_BENCHMARK(int8_t, rocprim::plus<int8_t>)
    CREATE_BENCHMARK(uint8_t, rocprim::plus<uint8_t>)
    CREATE_BENCHMARK(rocprim::half, rocprim::plus<rocprim::half>)

    CREATE_BENCHMARK(custom_float2, rocprim::plus<custom_float2>)
    CREATE_BENCHMARK(custom_double2, rocprim::plus<custom_double2>)
#endif

    // Use manual timing
    for(auto& b : benchmarks)
    {
        b->UseManualTime();
        b->Unit(benchmark::kMillisecond);
    }

    // Force number of iterations
    if(trials > 0)
    {
        for(auto& b : benchmarks)
        {
            b->Iterations(trials);
        }
    }

    // Run benchmarks
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}
