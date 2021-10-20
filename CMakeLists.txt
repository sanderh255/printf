cmake_minimum_required (VERSION 3.20)

include(CheckTypeSize)

project(
	printf
	LANGUAGES C
	DESCRIPTION "Mostly-stand-alone C implementation of printf, vprintf, sprintf and related functions"
	HOMEPAGE_URL https://github.com/eyalroz/printf
	VERSION 5.1.0
)

option(BUILD_FOR_CUDA "Built a version of the library for CUDA device-side use"  OFF)
if (BUILD_FOR_CUDA)
	enable_language(CUDA)
endif()


option(BUILD_TESTS          "Build test programs for the library" OFF)
option(BUILD_STATIC_LIBRARY "Build the library as static rather than shared" OFF)

# Boolean options which go into config.h

option(SUPPORT_DECIMAL_SPECIFIERS      "Support decimal notation floating-point conversion specifiers (%f,%F)" ON)
option(SUPPORT_EXPONENTIAL_SPECIFIERS  "Support exponential floating point format conversion specifiers (%e,%E,%g,%G)" ON)
option(SUPPORT_LONG_LONG               "Support long long integral types (allows for the ll length modifier and affects %p)" ON)
option(ALIAS_STANDARD_FUNCTION_NAMES   "Alias the standard library function names (printf, sprintf etc.) to the library's functions" ON)

foreach(opt
	SUPPORT_DECIMAL_SPECIFIERS
	SUPPORT_EXPONENTIAL_SPECIFIERS
	SUPPORT_LONG_LONG
	ALIAS_STANDARD_FUNCTION_NAMES)
	if (${${opt}})
		set("PRINTF_${opt}" 1)
	else()
		set("PRINTF_${opt}" 0)
		message(STATUS "PRINTF_${opt} 0")
	endif()
endforeach()

# Numeric defines which go into config.h

set(NTOA_BUFFER_SIZE "32" CACHE STRING          "Integer to string (ntoa) conversion buffer size")
set(PRINTF_DECIMAL_BUFFER_SIZE "32" CACHE STRING   "Floating-point to decimal conversion buffer size")
set(DEFAULT_FLOAT_PRECISION "6" CACHE STRING    "Default precision when printing floating-point values")
set(MAX_INTEGRAL_DIGITS_FOR_DECIMAL "9" CACHE STRING "Maximum number of integral-part digits of a floating-point value for which printing with %f uses decimal (non-exponential) notation")

# Checks related to the 'j', 'z' and 't' size modifiers

check_type_size( "long"        SIZEOF_LONG       )
check_type_size( "long long"   SIZEOF_LONG_LONG  )

set(ACCEPTABLE_JZT_TYPE_SIZES ${SIZEOF_LONG} ${SIZEOF_LONG_LONG})

function(validate_type_size type_name)
	check_type_size(${type_name} TYPE_SIZE)
	if (NOT ${TYPE_SIZE} IN_LIST ACCEPTABLE_JZT_TYPE_SIZES)
		message(FATAL_ERROR "sizeof(${type_name}) is  ${TYPE_SIZE}, which is neither sizeof(long) (${SIZEOF_LONG}) nor sizeof(long long) (${SIZEOF_LONG_LONG}). Please contact the library maintainers with your platform details.")
	endif()
endfunction()
validate_type_size("intmax_t")
validate_type_size("size_t")
validate_type_size("ptrdiff_t")

if (BUILD_STATIC_LIBRARY)
	add_library(printf STATIC)
else()
	add_library(printf SHARED)
endif()
set(built_targets printf)

if (BUILD_FOR_CUDA)
	add_library(printf_cuda STATIC)
	set_target_properties(printf_cuda PROPERTIES LINKER_LANGUAGE CUDA)
	list(APPEND built_targets printf_cuda)
endif()

set(GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")
configure_file("printf_config.h.in" "${GENERATED_INCLUDE_DIR}/printf_config.h" @ONLY)

foreach(tgt ${built_targets})
	target_sources(${tgt} PRIVATE "${GENERATED_INCLUDE_DIR}/printf_config.h" printf.h)
	target_compile_definitions(${tgt} PRIVATE PRINTF_INCLUDE_CONFIG_H)
	target_include_directories(${tgt}
		PRIVATE
			"$<BUILD_INTERFACE:${GENERATED_INCLUDE_DIR}>"
			${CMAKE_CURRENT_SOURCE_DIR}
		PRIVATE
		"$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
	)
	set_target_properties(${tgt} PROPERTIES
			 LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
			 ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
endforeach()

target_sources(printf PRIVATE printf.c)
set_target_properties(
	printf
	PROPERTIES
	C_STANDARD 99
	C_STANDARD_REQUIRED YES
	C_EXTENSIONS OFF
)

if (BUILD_FOR_CUDA)

	# This extra file is necessary for making CMake generate
	# the extra printf.cu file only when printf.c changes, rather than
	# every time.

	file(GENERATE
		OUTPUT generate-printf-cu.cmake
		CONTENT [[
cmake_minimum_required(VERSION 3.20)
file(WRITE "${OUT}" "#include \"printf.c\"")
]]
	)

	# This extra file is necessary, because CMake won't let us compile
	# printf.c once as a C file, then again as a CUDA file. See:
	# https://gitlab.kitware.com/cmake/cmake/-/issues/22779
	add_custom_command(
		OUTPUT printf.cu
		COMMAND ${CMAKE_COMMAND} -DOUT=printf.cu -P generate-printf-cu.cmake
		COMMENT "Generating printf.cu from printf.c"
		DEPENDS printf.c)

	set_property(SOURCE printf.cu PROPERTY LANGUAGE CUDA)

	#	add_dependencies(printf_cuda printf.cu)
	target_sources(printf_cuda PRIVATE printf.cu)
	set_target_properties(
		printf_cuda
		PROPERTIES
		CUDA_STANDARD_REQUIRED YES
		CUDA_EXTENSIONS OFF
		CUDA_SEPARABLE_COMPILATION ON
	)
endif()

if (CMAKE_C_COMPILER_ID STREQUAL "MSVC")
	target_compile_options(printf PRIVATE /W4)
elseif (CMAKE_C_COMPILER_ID STREQUAL "GNU" OR
	CMAKE_C_COMPILER_ID STREQUAL "Clang")
	target_compile_options(printf PRIVATE -Wall -Wextra -pedantic)
endif()
# TODO: Consider also enabling warnings with nvcc or other CUDA compilers

if (BUILD_TESTS)
	enable_testing()
	add_subdirectory(test)
endif()

if (UNIX)
	add_custom_target(printf-sizes
		COMMAND size -A -t $<TARGET_FILE:printf>  > printf_sizes.txt
		DEPENDS printf
		BYPRODUCTS printf_sizes.txt
		COMMENT Prints the sizes of the different sections of the ELF file: text, dat, vss etc.)

	add_custom_target(printf-symbols
		COMMAND nm --numeric-sort --print-size "$<TARGET_FILE:printf>" > printf_symbols.txt
		COMMAND bash -c "nm --numeric-sort --print-size $<TARGET_FILE:printf> | c++filt > printf_cpp_symbols.txt"
		VERBATIM
		DEPENDS printf
		BYPRODUCTS printf_symbols.txt printf_cpp_symbols.txt
		COMMENT Produces lists of the symbols, and C++demangled symbols, inside the library)

	add_custom_target(printf-lst
		COMMAND objdump --disassemble --line-numbers -S "$<TARGET_FILE:printf>" > printf.list
		DEPENDS printf
		BYPRODUCTS printf.lst
		COMMENT Dissassembles the compiled library into an .lst file)
endif()

# -------------------------
# Installation
# -------------------------

include(GNUInstallDirs)

install(
	TARGETS printf printf_cuda 
	EXPORT printf_export
	RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
	ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
	LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
	INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

install(
	FILES "printf.h"
	DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

install(
	EXPORT printf_export
	DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/printf"
	NAMESPACE "printf::"
	FILE "printf-config.cmake"
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
	"printf-config-version.cmake"
	VERSION ${PROJECT_VERSION}
	COMPATIBILITY SameMinorVersion
)

install(
	FILES "${CMAKE_CURRENT_BINARY_DIR}/printf-config-version.cmake"
	DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/printf"
)

