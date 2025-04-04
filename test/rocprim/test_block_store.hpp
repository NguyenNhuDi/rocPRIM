#include "../common_test_header.hpp"
#include "test_utils.hpp"

#include "../../common/utils.hpp"
#include "../../common/utils_device_ptr.hpp"
#include "test_seed.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_data_generation.hpp"
#include "test_utils_sort_comparator.hpp"

#include <rocprim/block/block_sort.hpp>
#include <rocprim/detail/various.hpp>
#include <rocprim/functional.hpp>
#include <rocprim/types/tuple.hpp>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdint.h>
#include <type_traits>
#include <utility>
#include <vector>
#include <random>

block_store_test_suite_type_def(suite_name, name_suffix);
typed_test_suite_def(suite_name, name_suffix, block_params);

// using header guards for these test functions because this file is included multiple times:
// once for the integrals test suite and once for the floating point test suite.
#ifndef TEST_ROCPRIM_TEST_BLOCK_STORE_HPP_
    #define TEST_ROCPRIM_TEST_BLOCK_STORE_HPP_

    template<
    unsigned int block_size,
    unsigned int items_per_thread,
    typename DataType,
    rocprim::block_store_method algorithm
>
void TestStore(){
    constexpr size_t items_per_block = block_size * items_per_thread;
    constexpr size_t grid_size = 1144;
    constexpr size_t size = items_per_block * grid_size;
    

    for(size_t seed_index = 0; seed_index < number_of_runs; seed_index++)
    {
        unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];

        std::vector<DataType> input_value(size);

        for(size_t i = 0; i < size; i++) input_value[i] = static_cast<DataType>(i);
             
        common::device_ptr<DataType> device_value_input(input_value);

        hipLaunchKernelGGL(HIP_KERNEL_NAME(store_kernel<
                                            block_size,
                                            items_per_thread,
                                            DataType,
                                            DataType*,
                                            algorithm
                                        >),
            dim3(grid_size),
            dim3(block_size),
            0,
            0,
            device_value_input.get()
        );

        HIP_CHECK(hipGetLastError());

        std::vector<DataType> output_value = device_value_input.load();

        test_utils::assert_eq(output_value, input_value);

    }
}

#endif
typed_test_def(suite_name, name_suffix, Store)
{
    using DataType                                                      = typename TestFixture::DataType;
    static constexpr const rocprim::block_store_method algo             = TEST_BLOCK_STORE_ALGORITHM;
    static constexpr const unsigned int                block_size       = TestFixture::block_size;
    static constexpr const unsigned int                items_per_thread = 1;
    TestStore<block_size, items_per_thread, DataType, algo>();
}
