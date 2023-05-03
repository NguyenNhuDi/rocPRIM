// Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCPRIM_DEVICE_DETAIL_CONFIG_DEVICE_REDUCE_HPP_
#define ROCPRIM_DEVICE_DETAIL_CONFIG_DEVICE_REDUCE_HPP_

#include "../../../type_traits.hpp"
#include "../device_config_helper.hpp"
#include <type_traits>

/* DO NOT EDIT THIS FILE
 * This file is automatically generated by `/scripts/autotune/create_optimization.py`.
 * so most likely you want to edit rocprim/device/device_(algo)_config.hpp
 */

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<unsigned int arch, class key_type, class enable = void>
struct default_reduce_config : default_reduce_config_base<key_type>
{};

// Based on key_type = double
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::gfx1030),
    key_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 8)
                      && (sizeof(key_type) > 4))>>
    : reduce_config<256, 8, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = float
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::gfx1030),
    key_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 4)
                      && (sizeof(key_type) > 2))>>
    : reduce_config<256, 2, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = rocprim::half
template<class key_type>
struct default_reduce_config<static_cast<unsigned int>(target_arch::gfx1030),
                             key_type,
                             std::enable_if_t<(bool(rocprim::is_floating_point<key_type>::value)
                                               && (sizeof(key_type) <= 2))>>
    : reduce_config<256, 4, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = int64_t
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::gfx1030),
    key_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 8)
                      && (sizeof(key_type) > 4))>>
    : reduce_config<256, 1, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = int
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::gfx1030),
    key_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 4)
                      && (sizeof(key_type) > 2))>>
    : reduce_config<256, 2, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = short
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::gfx1030),
    key_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 2)
                      && (sizeof(key_type) > 1))>>
    : reduce_config<256, 4, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = int8_t
template<class key_type>
struct default_reduce_config<static_cast<unsigned int>(target_arch::gfx1030),
                             key_type,
                             std::enable_if_t<(!bool(rocprim::is_floating_point<key_type>::value)
                                               && (sizeof(key_type) <= 1))>>
    : reduce_config<128, 16, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = double
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::gfx1102),
    key_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 8)
                      && (sizeof(key_type) > 4))>>
    : reduce_config<256, 8, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = float
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::gfx1102),
    key_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 4)
                      && (sizeof(key_type) > 2))>>
    : reduce_config<128, 16, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = rocprim::half
template<class key_type>
struct default_reduce_config<static_cast<unsigned int>(target_arch::gfx1102),
                             key_type,
                             std::enable_if_t<(bool(rocprim::is_floating_point<key_type>::value)
                                               && (sizeof(key_type) <= 2))>>
    : reduce_config<256, 16, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = int64_t
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::gfx1102),
    key_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 8)
                      && (sizeof(key_type) > 4))>>
    : reduce_config<128, 16, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = int
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::gfx1102),
    key_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 4)
                      && (sizeof(key_type) > 2))>>
    : reduce_config<256, 8, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = short
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::gfx1102),
    key_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 2)
                      && (sizeof(key_type) > 1))>>
    : reduce_config<256, 8, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = int8_t
template<class key_type>
struct default_reduce_config<static_cast<unsigned int>(target_arch::gfx1102),
                             key_type,
                             std::enable_if_t<(!bool(rocprim::is_floating_point<key_type>::value)
                                               && (sizeof(key_type) <= 1))>>
    : reduce_config<256, 16, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = double
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::gfx900),
    key_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 8)
                      && (sizeof(key_type) > 4))>>
    : reduce_config<128, 4, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = float
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::gfx900),
    key_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 4)
                      && (sizeof(key_type) > 2))>>
    : reduce_config<128, 8, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = rocprim::half
template<class key_type>
struct default_reduce_config<static_cast<unsigned int>(target_arch::gfx900),
                             key_type,
                             std::enable_if_t<(bool(rocprim::is_floating_point<key_type>::value)
                                               && (sizeof(key_type) <= 2))>>
    : reduce_config<128, 16, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = int64_t
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::gfx900),
    key_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 8)
                      && (sizeof(key_type) > 4))>>
    : reduce_config<128, 4, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = int
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::gfx900),
    key_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 4)
                      && (sizeof(key_type) > 2))>>
    : reduce_config<128, 8, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = short
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::gfx900),
    key_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 2)
                      && (sizeof(key_type) > 1))>>
    : reduce_config<128, 16, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = int8_t
template<class key_type>
struct default_reduce_config<static_cast<unsigned int>(target_arch::gfx900),
                             key_type,
                             std::enable_if_t<(!bool(rocprim::is_floating_point<key_type>::value)
                                               && (sizeof(key_type) <= 1))>>
    : reduce_config<64, 16, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = double
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::gfx906),
    key_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 8)
                      && (sizeof(key_type) > 4))>>
    : reduce_config<128, 4, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = float
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::gfx906),
    key_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 4)
                      && (sizeof(key_type) > 2))>>
    : reduce_config<256, 4, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = rocprim::half
template<class key_type>
struct default_reduce_config<static_cast<unsigned int>(target_arch::gfx906),
                             key_type,
                             std::enable_if_t<(bool(rocprim::is_floating_point<key_type>::value)
                                               && (sizeof(key_type) <= 2))>>
    : reduce_config<64, 16, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = int64_t
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::gfx906),
    key_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 8)
                      && (sizeof(key_type) > 4))>>
    : reduce_config<256, 16, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = int
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::gfx906),
    key_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 4)
                      && (sizeof(key_type) > 2))>>
    : reduce_config<256, 4, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = short
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::gfx906),
    key_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 2)
                      && (sizeof(key_type) > 1))>>
    : reduce_config<256, 8, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = int8_t
template<class key_type>
struct default_reduce_config<static_cast<unsigned int>(target_arch::gfx906),
                             key_type,
                             std::enable_if_t<(!bool(rocprim::is_floating_point<key_type>::value)
                                               && (sizeof(key_type) <= 1))>>
    : reduce_config<128, 16, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = double
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::gfx908),
    key_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 8)
                      && (sizeof(key_type) > 4))>>
    : reduce_config<256, 16, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = float
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::gfx908),
    key_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 4)
                      && (sizeof(key_type) > 2))>>
    : reduce_config<256, 4, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = rocprim::half
template<class key_type>
struct default_reduce_config<static_cast<unsigned int>(target_arch::gfx908),
                             key_type,
                             std::enable_if_t<(bool(rocprim::is_floating_point<key_type>::value)
                                               && (sizeof(key_type) <= 2))>>
    : reduce_config<256, 16, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = int64_t
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::gfx908),
    key_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 8)
                      && (sizeof(key_type) > 4))>>
    : reduce_config<256, 16, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = int
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::gfx908),
    key_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 4)
                      && (sizeof(key_type) > 2))>>
    : reduce_config<256, 4, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = short
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::gfx908),
    key_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 2)
                      && (sizeof(key_type) > 1))>>
    : reduce_config<128, 8, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = int8_t
template<class key_type>
struct default_reduce_config<static_cast<unsigned int>(target_arch::gfx908),
                             key_type,
                             std::enable_if_t<(!bool(rocprim::is_floating_point<key_type>::value)
                                               && (sizeof(key_type) <= 1))>>
    : reduce_config<256, 16, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = double
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::unknown),
    key_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 8)
                      && (sizeof(key_type) > 4))>>
    : reduce_config<256, 16, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = float
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::unknown),
    key_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 4)
                      && (sizeof(key_type) > 2))>>
    : reduce_config<256, 4, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = rocprim::half
template<class key_type>
struct default_reduce_config<static_cast<unsigned int>(target_arch::unknown),
                             key_type,
                             std::enable_if_t<(bool(rocprim::is_floating_point<key_type>::value)
                                               && (sizeof(key_type) <= 2))>>
    : reduce_config<256, 16, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = int64_t
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::unknown),
    key_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 8)
                      && (sizeof(key_type) > 4))>>
    : reduce_config<256, 16, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = int
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::unknown),
    key_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 4)
                      && (sizeof(key_type) > 2))>>
    : reduce_config<256, 4, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = short
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::unknown),
    key_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 2)
                      && (sizeof(key_type) > 1))>>
    : reduce_config<128, 8, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = int8_t
template<class key_type>
struct default_reduce_config<static_cast<unsigned int>(target_arch::unknown),
                             key_type,
                             std::enable_if_t<(!bool(rocprim::is_floating_point<key_type>::value)
                                               && (sizeof(key_type) <= 1))>>
    : reduce_config<256, 16, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = double
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::gfx90a),
    key_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 8)
                      && (sizeof(key_type) > 4))>>
    : reduce_config<256, 16, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = float
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::gfx90a),
    key_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 4)
                      && (sizeof(key_type) > 2))>>
    : reduce_config<256, 4, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = rocprim::half
template<class key_type>
struct default_reduce_config<static_cast<unsigned int>(target_arch::gfx90a),
                             key_type,
                             std::enable_if_t<(bool(rocprim::is_floating_point<key_type>::value)
                                               && (sizeof(key_type) <= 2))>>
    : reduce_config<256, 16, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = int64_t
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::gfx90a),
    key_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 8)
                      && (sizeof(key_type) > 4))>>
    : reduce_config<256, 16, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = int
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::gfx90a),
    key_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 4)
                      && (sizeof(key_type) > 2))>>
    : reduce_config<256, 4, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = short
template<class key_type>
struct default_reduce_config<
    static_cast<unsigned int>(target_arch::gfx90a),
    key_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) <= 2)
                      && (sizeof(key_type) > 1))>>
    : reduce_config<128, 8, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

// Based on key_type = int8_t
template<class key_type>
struct default_reduce_config<static_cast<unsigned int>(target_arch::gfx90a),
                             key_type,
                             std::enable_if_t<(!bool(rocprim::is_floating_point<key_type>::value)
                                               && (sizeof(key_type) <= 1))>>
    : reduce_config<256, 16, ::rocprim::block_reduce_algorithm::using_warp_reduce>
{};

} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DETAIL_CONFIG_DEVICE_REDUCE_HPP_