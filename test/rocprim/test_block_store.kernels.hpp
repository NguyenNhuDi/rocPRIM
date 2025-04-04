#ifndef TEST_BLOCK_STORE_KERNELS_HPP_
#define TEST_BLOCK_STORE_KERNELS_HPP_

constexpr bool is_buildable(unsigned int                BlockSize,
    unsigned int                ItemsPerThread,
    rocprim::block_store_method algorithm)
{
    switch(algorithm)
    {
        case rocprim::block_store_method::block_store_direct: 
        case rocprim::block_store_method::block_store_striped: 
            return true;
        case rocprim::block_store_method::block_store_vectorize:
            return (ItemsPerThread % 2 == 0) && ((BlockSize * ItemsPerThread) % 4 == 0);
    }
    return false;
}
template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    typename DataType,
    class OutputIterator,
    rocprim::block_store_method algorithm,
    std::enable_if_t<(ItemsPerThread == 1u && is_buildable(BlockSize, ItemsPerThread, algorithm)),
        int> = 0
    >
    __global__ __launch_bounds__(BlockSize) void store_kernel(DataType * input){
        using bstore_type = rocprim::block_store<DataType, BlockSize, ItemsPerThread, algorithm>;

        static constexpr const unsigned int ItemsPerBlock = ItemsPerThread * BlockSize;
        const unsigned int                  block_offset  = blockIdx.x * ItemsPerBlock;
        const unsigned int                  index         = block_offset + (threadIdx.x * ItemsPerThread);

        DataType temp[ItemsPerThread];
        DataType storage[ItemsPerThread];

        for(size_t i = 0; i < ItemsPerThread; i++)
            temp[i] = input[index + i];

        bstore_type().store(temp, storage);

        for(size_t i = 0; i < ItemsPerThread; i++)
            input[index + i] = storage[i];

    }

    template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread,
    typename DataType,
    class OutputIterator,
    rocprim::block_store_method algorithm,
    std::enable_if_t<!(ItemsPerThread == 1u && is_buildable(BlockSize, ItemsPerThread, algorithm)),
        int> = 0
    >
    __global__ __launch_bounds__(BlockSize) void store_kernel(DataType * input){}

#endif