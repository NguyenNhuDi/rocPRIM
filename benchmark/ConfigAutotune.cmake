# MIT License
#
# Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

include(CMakePrintHelpers)
option(BENCHMARK_CONFIG_AUTOTUNE "Benchmark device-level functions using various configs" OFF)

function(add_configured_source)
  cmake_parse_arguments(PARSE_ARGV 0 ARG "" "INPUT;TARGET;OUTPUT_PATTERN" "NAMES;VALUES")
  list(LENGTH ARG_NAMES NAMES_LEN)
  list(LENGTH ARG_LISTS VALS_LEN)
  if (NOT NAMES_LEN EQUAL VALS_LEN)
    message("NAMES_LEN: ${NAMES_LEN}, VALS_LEN: ${VALS_LEN}")
    message(FATAL_ERROR "The same number of names and values must be provided!")
  endif()

  math(EXPR max "${VALS_LEN} - 1")
  foreach(i RANGE ${max})
    list(GET ARG_NAMES ${i} curr_name)
    list(GET ARG_VALUES ${i} "${curr_name}")
  endforeach()

  string(CONFIGURE "${ARG_OUTPUT_PATTERN}" output @ONLY)
  string(REPLACE ":" "_" output ${output})
  configure_file("${ARG_INPUT}" "${ARG_TARGET}.parallel/${output}" @ONLY)
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES "${ARG_TARGET}.parallel")
  target_sources("${ARG_TARGET}" PRIVATE "${ARG_TARGET}.parallel/${output}")
  target_include_directories("${ARG_TARGET}" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

  # Cmake configuration needs to be rerun if the input template changes
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${ARG_INPUT}")
endfunction()

function(div_round_up dividend divisor result_var)
  math(EXPR result "(${dividend} + ${divisor} - 1) / ${divisor}")
  set("${result_var}" "${result}" PARENT_SCOPE)
endfunction()

function(read_config_autotune_settings file list_across_names list_across output_pattern_suffix)
  file(READ ${file} FILE_CONTENTS)
  string(REGEX MATCHALL "CONFIG_AUTOTUNE_[A-z]*=[^\$]*" regex_found_list ${FILE_CONTENTS})
  foreach(match IN LISTS regex_found_list)
    string(REGEX MATCH "CONFIG_AUTOTUNE_([A-z]*)=(.*)" _ ${match})
    list(APPEND across_names ${CMAKE_MATCH_1})
    list(APPEND across ${CMAKE_MATCH_2})
  endforeach()
  set(list_across_names ${across_names} PARENT_SCOPE)
  set(list_across ${across} PARENT_SCOPE)
  string(REGEX MATCH "CA_OUTPUT_PATTERN_SUFFIX=([^\$]*)" _ ${FILE_CONTENTS})
  set(output_pattern_suffix ${CMAKE_MATCH_1} PARENT_SCOPE)
endfunction()

function(add_matrix)
  set(single_value_args "TARGET" "INPUT" "OUTPUT_PATTERN" "SHARDS" "CURRENT_SHARD")
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
    list(APPEND single_value_args "FILTER")
  endif()
  cmake_parse_arguments(PARSE_ARGV 0 ARG "" "${single_value_args}" "NAMES;LISTS")
  list(LENGTH ARG_NAMES NAMES_LEN)
  list(LENGTH ARG_LISTS LISTS_LEN)
  if (NOT NAMES_LEN EQUAL LISTS_LEN)
    message("NAMES_LEN: ${NAMES_LEN}, LISTS_LEN: ${LISTS_LEN}")
    message(FATAL_ERROR "The same number of names and lists must be provided!")
  endif()

  # Calculate the total number of permutations
  set(total_len 1)
  foreach(LIST IN LISTS ARG_LISTS)
    string(REPLACE " " ";" list ${LIST})
    list(LENGTH list LIST_LEN)
    math(EXPR total_len "${total_len} * ${LIST_LEN}")
  endforeach()

  if(NOT DEFINED ARG_SHARDS)
    set(ARG_SHARDS 1)
  endif()
  div_round_up("${total_len}" "${ARG_SHARDS}" per_shard)
  message("per_shard: ${per_shard}")
  math(EXPR start "${ARG_CURRENT_SHARD} * ${per_shard}")
  math(EXPR stop "${start} + ${per_shard} - 1")
  message("start: ${start}, stop: ${stop}")
  # Run for each permutation of input paramters
  foreach(i RANGE ${start} ${stop})
    set(index ${i})
    set(values "")
    foreach(input_list IN LISTS ARG_LISTS)
      string(REPLACE " " ";" curr_list ${input_list})
      list(LENGTH curr_list curr_length)
      math(EXPR curr_index "${index} % ${curr_length}")

      list(GET curr_list ${curr_index} curr_item)
      list(APPEND values "${curr_item}")
      math(EXPR index "${index} / ${curr_length}")
    endforeach()

    if(DEFINED ARG_FILTER)
      cmake_language(CALL "${ARG_FILTER}" keep ${values})
      if(NOT keep)
        continue()
      endif()
    endif()

    add_configured_source(TARGET "${ARG_TARGET}"
            INPUT "${ARG_INPUT}"
            OUTPUT_PATTERN "${ARG_OUTPUT_PATTERN}"
            NAMES ${ARG_NAMES}
            VALUES ${values})
  endforeach()
endfunction()

function(reject_all RESULT)
  set("${RESULT}" OFF PARENT_SCOPE)
endfunction()

function(accept_all RESULT)
  set("${RESULT}" ON PARENT_SCOPE)
endfunction()

function(reject_odd_blocksize RESULT BlockSize)
  math(EXPR res "${BlockSize} % 2")
  if(res EQUAL 0)
    set("${RESULT}" ON PARENT_SCOPE)
  else()
    set("${RESULT}" OFF PARENT_SCOPE)
  endif()
endfunction()
