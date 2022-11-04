// MIT License
//
// Copyright (c) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BENCHMARK_DEVICE_RADIX_SORT_PARALLEL_HPP_
#define ROCPRIM_BENCHMARK_DEVICE_RADIX_SORT_PARALLEL_HPP_

#include <cstddef>
#include <string>
#include <vector>

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

// rocPRIM
#include <rocprim/rocprim.hpp>

#include "benchmark_utils.hpp"

namespace rp = rocprim;

template<typename Key   = int,
         typename Value = rocprim::empty_type,
         typename Config
         = rocprim::detail::default_radix_sort_config<ROCPRIM_TARGET_ARCH, Key, Value>>
struct device_radix_sort_benchmark : public config_autotune_interface
{
    static std::string get_name_pattern()
    {
        return R"regex((?P<algo>\S*?)<)regex"
               R"regex((?P<key_type>\S*),(?:\s*(?P<value_type>\S*),)?\s*radix_sort_config<)regex"
               R"regex((?P<long_radix_bits>[0-9]+),\s*(?P<short_radix_bits>[0-9]+),\s*)regex"
               R"regex(kernel_config<\s*(?P<scan_block_size>[0-9]+),\s*(?P<scan_items_per_thread>[0-9]+)>,\s*)regex"
               R"regex(kernel_config<\s*(?P<sort_block_size>[0-9]+),\s*(?P<sort_items_per_thread>[0-9]+)>,\s*)regex"
               R"regex(kernel_config<\s*(?P<sort_single_block_size>[0-9]+),\s*(?P<sort_single_items_per_thread>[0-9]+)>,\s*)regex"
               R"regex(kernel_config<\s*(?P<sort_merge_block_size>[0-9]+),\s*(?P<sort_merge_items_per_thread>[0-9]+)>,\s*)regex"
               R"regex((?P<force_single_kernel_config>[0-9]+)>>)regex";
    }

    std::string name() const override
    {
        using namespace std::string_literals;
        return std::string(
            "device_radix_sort<" + std::string(Traits<Key>::name()) + ", "
            + (std::is_same<Value, rocprim::empty_type>::value
                   ? ""s
                   : std::string(Traits<Value>::name()) + ", ")
            + "radix_sort_config<" + std::to_string(Config::long_radix_bits) + ", "
            + std::to_string(Config::short_radix_bits) + ", " + "kernel_config<"
            + pad_string(std::to_string(Config::scan::block_size), 4) + ", "
            + pad_string(std::to_string(Config::scan::items_per_thread), 2) + ">, "
            + "kernel_config<" + pad_string(std::to_string(Config::sort::block_size), 4) + ", "
            + pad_string(std::to_string(Config::sort::items_per_thread), 2) + ">, "
            + "kernel_config<" + pad_string(std::to_string(Config::sort_single::block_size), 4)
            + ", " + pad_string(std::to_string(Config::sort_single::items_per_thread), 2) + ">, "
            + "kernel_config<" + pad_string(std::to_string(Config::sort_merge::block_size), 4)
            + ", " + pad_string(std::to_string(Config::sort_merge::items_per_thread), 2) + ">, "
            + std::to_string(Config::force_single_kernel_config) + ">>");
    }

    static constexpr unsigned int batch_size  = 10;
    static constexpr unsigned int warmup_size = 5;

    static std::vector<Key> generate_keys(size_t size)
    {
        using key_type = Key;

        if(std::is_floating_point<key_type>::value)
        {
            return get_random_data<key_type>(size, (key_type)-1000, (key_type) + 1000, size);
        }
        else
        {
            return get_random_data<key_type>(size,
                                             std::numeric_limits<key_type>::min(),
                                             std::numeric_limits<key_type>::max(),
                                             size);
        }
    }

    // keys benchmark
    template<typename val = Value>
    auto do_run(benchmark::State& state, size_t size, const hipStream_t stream) const ->
        typename std::enable_if<std::is_same<val, ::rocprim::empty_type>::value, void>::type
    {
        auto keys_input = generate_keys(size);

        using key_type = Key;

        key_type* d_keys_input;
        key_type* d_keys_output;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_keys_input), size * sizeof(key_type)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_keys_output), size * sizeof(key_type)));

        HIP_CHECK(hipMemcpy(d_keys_input,
                            keys_input.data(),
                            size * sizeof(key_type),
                            hipMemcpyHostToDevice));

        void*  d_temporary_storage     = nullptr;
        size_t temporary_storage_bytes = 0;
        HIP_CHECK(rp::radix_sort_keys<Config>(d_temporary_storage,
                                              temporary_storage_bytes,
                                              d_keys_input,
                                              d_keys_output,
                                              size,
                                              0,
                                              sizeof(key_type) * 8,
                                              stream,
                                              false));

        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        // Warm-up
        for(size_t i = 0; i < warmup_size; i++)
        {
            HIP_CHECK(rp::radix_sort_keys<Config>(d_temporary_storage,
                                                  temporary_storage_bytes,
                                                  d_keys_input,
                                                  d_keys_output,
                                                  size,
                                                  0,
                                                  sizeof(key_type) * 8,
                                                  stream,
                                                  false));
        }
        HIP_CHECK(hipDeviceSynchronize());

        // HIP events creation
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        for(auto _ : state)
        {
            // Record start event
            HIP_CHECK(hipEventRecord(start, stream));

            for(size_t i = 0; i < batch_size; i++)
            {
                HIP_CHECK(rp::radix_sort_keys<Config>(d_temporary_storage,
                                                      temporary_storage_bytes,
                                                      d_keys_input,
                                                      d_keys_output,
                                                      size,
                                                      0,
                                                      sizeof(key_type) * 8,
                                                      stream,
                                                      false));
            }

            // Record stop event and wait until it completes
            HIP_CHECK(hipEventRecord(stop, stream));
            HIP_CHECK(hipEventSynchronize(stop));

            float elapsed_mseconds;
            HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
            state.SetIterationTime(elapsed_mseconds / 1000);
        }

        // Destroy HIP events
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));

        state.SetBytesProcessed(state.iterations() * batch_size * size * sizeof(key_type));
        state.SetItemsProcessed(state.iterations() * batch_size * size);

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_keys_input));
        HIP_CHECK(hipFree(d_keys_output));
    }

    // pairs benchmark
    template<typename val = Value>
    auto do_run(benchmark::State& state, size_t size, const hipStream_t stream) const ->
        typename std::enable_if<!std::is_same<val, ::rocprim::empty_type>::value, void>::type
    {
        auto keys_input = generate_keys(size);

        using key_type   = Key;
        using value_type = Value;

        std::vector<value_type> values_input(size);
        for(size_t i = 0; i < size; i++)
        {
            values_input[i] = value_type(i);
        }

        key_type* d_keys_input;
        key_type* d_keys_output;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_keys_input), size * sizeof(key_type)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_keys_output), size * sizeof(key_type)));
        HIP_CHECK(hipMemcpy(d_keys_input,
                            keys_input.data(),
                            size * sizeof(key_type),
                            hipMemcpyHostToDevice));

        value_type* d_values_input;
        value_type* d_values_output;
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_values_input), size * sizeof(value_type)));
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&d_values_output), size * sizeof(value_type)));
        HIP_CHECK(hipMemcpy(d_values_input,
                            values_input.data(),
                            size * sizeof(value_type),
                            hipMemcpyHostToDevice));

        void*  d_temporary_storage     = nullptr;
        size_t temporary_storage_bytes = 0;
        HIP_CHECK(rp::radix_sort_pairs<Config>(d_temporary_storage,
                                               temporary_storage_bytes,
                                               d_keys_input,
                                               d_keys_output,
                                               d_values_input,
                                               d_values_output,
                                               size,
                                               0,
                                               sizeof(key_type) * 8,
                                               stream,
                                               false));

        HIP_CHECK(hipMalloc(&d_temporary_storage, temporary_storage_bytes));
        HIP_CHECK(hipDeviceSynchronize());

        // Warm-up
        for(size_t i = 0; i < warmup_size; i++)
        {
            HIP_CHECK(rp::radix_sort_pairs<Config>(d_temporary_storage,
                                                   temporary_storage_bytes,
                                                   d_keys_input,
                                                   d_keys_output,
                                                   d_values_input,
                                                   d_values_output,
                                                   size,
                                                   0,
                                                   sizeof(key_type) * 8,
                                                   stream,
                                                   false));
        }
        HIP_CHECK(hipDeviceSynchronize());

        // HIP events creation
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        for(auto _ : state)
        {
            // Record start event
            HIP_CHECK(hipEventRecord(start, stream));

            for(size_t i = 0; i < batch_size; i++)
            {
                HIP_CHECK(rp::radix_sort_pairs<Config>(d_temporary_storage,
                                                       temporary_storage_bytes,
                                                       d_keys_input,
                                                       d_keys_output,
                                                       d_values_input,
                                                       d_values_output,
                                                       size,
                                                       0,
                                                       sizeof(key_type) * 8,
                                                       stream,
                                                       false));
            }

            // Record stop event and wait until it completes
            HIP_CHECK(hipEventRecord(stop, stream));
            HIP_CHECK(hipEventSynchronize(stop));

            float elapsed_mseconds;
            HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
            state.SetIterationTime(elapsed_mseconds / 1000);
        }

        // Destroy HIP events
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));

        state.SetBytesProcessed(state.iterations() * batch_size * size
                                * (sizeof(key_type) + sizeof(value_type)));
        state.SetItemsProcessed(state.iterations() * batch_size * size);

        HIP_CHECK(hipFree(d_temporary_storage));
        HIP_CHECK(hipFree(d_keys_input));
        HIP_CHECK(hipFree(d_keys_output));
        HIP_CHECK(hipFree(d_values_input));
        HIP_CHECK(hipFree(d_values_output));
    }

    void run(benchmark::State& state, size_t size, hipStream_t stream) const override
    {
        do_run(state, size, stream);
    }
};

#ifdef BENCHMARK_CONFIG_TUNING

inline constexpr unsigned int get_max_items_per_thread(size_t key_bytes, size_t value_bytes)
{
    size_t total_bytes = key_bytes + value_bytes;
    if(total_bytes <= 12)
    {
        return 30;
    }
    else if(12 < total_bytes && total_bytes <= 18)
    {
        return 20;
    }
    else //(18 < total_bytes)
    {
        return 10;
    }
}
template<unsigned int LongRadixBits,
         unsigned int ShortRadixBits,
         unsigned int ItemsPerThread2,
         typename Key,
         typename Value = rocprim::empty_type>
struct device_radix_sort_benchmark_generator
{
    template<unsigned int ItemsPerThread1>
    struct create_ipt1
    {
        void operator()(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
        {
            storage.emplace_back(
                std::make_unique<device_radix_sort_benchmark<
                    Key,
                    Value,
                    rocprim::radix_sort_config<LongRadixBits,
                                               ShortRadixBits,
                                               rocprim::kernel_config<256u, ItemsPerThread1>,
                                               rocprim::kernel_config<256u, ItemsPerThread2>>>>());
        }
    };

    static void create(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
    {
        static constexpr unsigned int max_items_per_thread
            = get_max_items_per_thread(sizeof(Key), sizeof(Value));
        static_for_each<make_index_range<unsigned int, 1, max_items_per_thread>, create_ipt1>(
            storage);
    }
};

template<unsigned int BlockSize, typename Key, typename Value = rocprim::empty_type>
struct device_radix_sort_single_benchmark_generator
{
    template<unsigned int ItemsPerThread>
    struct create_ipt
    {
        void operator()(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
        {
            storage.emplace_back(
                std::make_unique<device_radix_sort_benchmark<
                    Key,
                    Value,
                    rocprim::radix_sort_config<8,
                                               7,
                                               rocprim::kernel_config<256, 2>,
                                               rocprim::kernel_config<256, 10>,
                                               rocprim::kernel_config<BlockSize, ItemsPerThread>,
                                               rocprim::kernel_config<1024, 1>,
                                               1024,
                                               true>>>());
        }
    };

    static void create(std::vector<std::unique_ptr<config_autotune_interface>>& storage)
    {
        static constexpr unsigned int max_items_per_thread
            = BlockSize < 512 ? get_max_items_per_thread(sizeof(Key), sizeof(Value))
                              : (BlockSize < 1024 ? 7 : 1);

        static_for_each<make_index_range<unsigned int, 1, max_items_per_thread>, create_ipt>(
            storage);
    }
};

#else // BENCHMARK_CONFIG_TUNING

    #define CREATE_RADIX_SORT_BENCHMARK(...)                         \
        {                                                            \
            const device_radix_sort_benchmark<__VA_ARGS__> instance; \
            REGISTER_BENCHMARK(benchmarks, size, stream, instance);  \
        }

inline void add_sort_keys_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                                     hipStream_t                                   stream,
                                     size_t                                        size)
{
    CREATE_RADIX_SORT_BENCHMARK(int)
    CREATE_RADIX_SORT_BENCHMARK(float)
    CREATE_RADIX_SORT_BENCHMARK(long long)
    CREATE_RADIX_SORT_BENCHMARK(int8_t)
    CREATE_RADIX_SORT_BENCHMARK(uint8_t)
    CREATE_RADIX_SORT_BENCHMARK(rocprim::half)
    CREATE_RADIX_SORT_BENCHMARK(short)
}

inline void add_sort_pairs_benchmarks(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                                      hipStream_t                                   stream,
                                      size_t                                        size)
{
    using custom_float2  = custom_type<float, float>;
    using custom_double2 = custom_type<double, double>;

    CREATE_RADIX_SORT_BENCHMARK(int, float)
    CREATE_RADIX_SORT_BENCHMARK(int, double)
    CREATE_RADIX_SORT_BENCHMARK(int, float2)
    CREATE_RADIX_SORT_BENCHMARK(int, custom_float2)
    CREATE_RADIX_SORT_BENCHMARK(int, double2)
    CREATE_RADIX_SORT_BENCHMARK(int, custom_double2)

    CREATE_RADIX_SORT_BENCHMARK(long long, float)
    CREATE_RADIX_SORT_BENCHMARK(long long, double)
    CREATE_RADIX_SORT_BENCHMARK(long long, float2)
    CREATE_RADIX_SORT_BENCHMARK(long long, custom_float2)
    CREATE_RADIX_SORT_BENCHMARK(long long, double2)
    CREATE_RADIX_SORT_BENCHMARK(long long, custom_double2)
    CREATE_RADIX_SORT_BENCHMARK(int8_t, int8_t)
    CREATE_RADIX_SORT_BENCHMARK(uint8_t, uint8_t)
    CREATE_RADIX_SORT_BENCHMARK(rocprim::half, rocprim::half)
}

#endif // BENCHMARK_CONFIG_TUNING

#endif // ROCPRIM_BENCHMARK_DEVICE_RADIX_SORT_PARALLEL_HPP_
