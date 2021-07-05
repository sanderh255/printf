cmake_minimum_required(VERSION 3.18)

enable_language(CXX)

option(BUILD_CUDA_TESTS "Include tests for use of the printf library with CUDA host-side and device-side code"  OFF)

set(test_targets test_suite autotest)

set(targets_needing_config_h test_suite)

option(TEST_WITH_NON_STANDARD_FORMAT_STRINGS "Include tests using non-standard-compliant format strings?" ON)
# ... don't worry, we'll suppress the compiler warnings for those.

if(BUILD_CUDA_TESTS)
	enable_language(CUDA)
	list(APPEND CMAKE_CUDA_FLAGS "--extended-lambda --expt-relaxed-constexpr -Xcudafe --display_error_number")
	if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
	    cmake_policy(SET CMP0104 OLD)
	endif()
	include(FindCUDA/select_compute_arch)
	list(APPEND cuda_test_targets cuda_test_suite_host cuda_test_suite_device)
	if((NOT DEFINED CUDA_ARCH_FLAGS) OR ("${CUDA_ARCH_FLAGS}" STREQUAL ""))
		cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS_1 Auto)
		set(CUDA_ARCH_FLAGS ${CUDA_ARCH_FLAGS} CACHE STRING "CUDA -gencode parameters")
		string(REPLACE ";" " " CUDA_ARCH_FLAGS_STR "${CUDA_ARCH_FLAGS}")
	else()
		set(CUDA_ARCH_FLAGS_STR "${CUDA_ARCH_FLAGS}")
	endif()
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_ARCH_FLAGS_STR}")

	foreach(tgt ${cuda_test_targets})
		string(REPLACE "cuda_" "cuda/" source_file_prefix ${tgt})
		add_executable(${tgt} "${source_file_prefix}.cu")
		target_include_directories(${tgt} PRIVATE "$<BUILD_INTERFACE:${GENERATED_INCLUDE_DIR}>")
		set_target_properties(
			${tgt}
			PROPERTIES
			CUDA_STANDARD 11
			CUDA_STANDARD_REQUIRED YES
			CUDA_EXTENSIONS NO
		)
		if (TEST_WITH_NON_STANDARD_FORMAT_STRINGS)
			target_compile_definitions(${tgt} PRIVATE TEST_WITH_NON_STANDARD_FORMAT_STRINGS)
		endif()
	endforeach()
endif()

if(BUILD_CUDA_TESTS)
	target_link_libraries(cuda_test_suite_device printf_cuda)
	set_target_properties(cuda_test_suite_device PROPERTIES CUDA_SEPARABLE_COMPILATION YES)
	list(APPEND targets_needing_config_h cuda_test_suite_host)
endif()

foreach(tgt ${test_targets})
	add_executable(${tgt} "${tgt}.cpp")
	set_target_properties(
		${tgt}
		PROPERTIES
		CXX_STANDARD 11
		CXX_STANDARD_REQUIRED YES
		CXX_EXTENSIONS NO
	)

    if (TEST_WITH_NON_STANDARD_FORMAT_STRINGS)
        target_compile_definitions(${tgt} PRIVATE TEST_WITH_NON_STANDARD_FORMAT_STRINGS)
    endif()

	if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
		target_compile_options(test_suite PRIVATE /W4)
	elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
			CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
		# lots of warnings and all warnings as errors
		target_compile_options(${tgt} PRIVATE
			-g
			-Wall
			-Wextra
			-pedantic
			-Wundef
			-Wsign-conversion
			-Wuninitialized
			-Wshadow
			-Wunreachable-code
			-Wswitch-default
			-Wswitch
			-Wcast-align
			-Wmissing-include-dirs
			-Winit-self
			-Wdouble-promotion
			-gdwarf-2
			-fno-exceptions
			-ffunction-sections
			-fdata-sections
			-fverbose-asm
			-Wunused-parameter
		)
		if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
			target_compile_options(${tgt} PRIVATE -ffat-lto-objects)
		endif()
	endif()
endforeach()

# These two lines are necessary, since the test suite does not actually use the
# compiled library - it includes the library's source .c file; and that means we
# need to include the generated config.h file.
foreach(tgt ${targets_needing_config_h})
	target_compile_definitions(${tgt} PRIVATE PRINTF_INCLUDE_CONFIG_H)
	target_include_directories(${tgt} PRIVATE "${GENERATED_INCLUDE_DIR}")
endforeach()
add_test(
	NAME "${PROJECT_NAME}.test_suite"
	COMMAND "test_suite" # ${TEST_RUNNER_PARAMS}
)

target_link_libraries(autotest PRIVATE printf)
target_include_directories(autotest PRIVATE "$<BUILD_INTERFACE:${GENERATED_INCLUDE_DIR}>")

# Not running autotest by default - it's randomized after all.

