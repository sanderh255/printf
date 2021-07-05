// #define CATCH_CONFIG_MAIN
// #include "catch.hpp"

#include "printf_config.h"
#include "../../printf.h"

#include <string.h>
#include <iostream>
#include <sstream>
#include <memory>
#include <math.h>

// Multi-compiler-compatible local warning suppression

#if defined(_MSC_VER)
  #define DISABLE_WARNING_PUSH           __pragma(warning( push ))
  #define DISABLE_WARNING_POP            __pragma(warning( pop ))
  #define DISABLE_WARNING(warningNumber) __pragma(warning( disable : warningNumber ))

  // TODO: find the right warning number for this
  #define DISABLE_WARNING_PRINTF_FORMAT             
  #define DISABLE_WARNING_PRINTF_FORMAT_EXTRA_ARGS  
  #define DISABLE_WARNING_PRINTF_FORMAT_OVERFLOW    

#elif defined(__NVCC__)
  #define DO_PRAGMA(X) _Pragma(#X)
  #define DISABLE_WARNING_PUSH           DO_PRAGMA(push)
  #define DISABLE_WARNING_POP            DO_PRAGMA(pop)
  #define DISABLE_WARNING(warning_code)  DO_PRAGMA(diag_suppress warning_code)

  #define DISABLE_WARNING_PRINTF_FORMAT             DISABLE_WARNING(bad_printf_format_string)
  #define DISABLE_WARNING_PRINTF_FORMAT_EXTRA_ARGS 
  #define DISABLE_WARNING_PRINTF_FORMAT_OVERFLOW 

#elif defined(__GNUC__) || defined(__clang__)
  #define DO_PRAGMA(X) _Pragma(#X)
  #define DISABLE_WARNING_PUSH           DO_PRAGMA(GCC diagnostic push)
  #define DISABLE_WARNING_POP            DO_PRAGMA(GCC diagnostic pop)
  #define DISABLE_WARNING(warningName)   DO_PRAGMA(GCC diagnostic ignored #warningName)

  #define DISABLE_WARNING_PRINTF_FORMAT             DISABLE_WARNING(-Wformat)
  #define DISABLE_WARNING_PRINTF_FORMAT_EXTRA_ARGS  DISABLE_WARNING(-Wformat-extra-args)
   #define DISABLE_WARNING_PRINTF_FORMAT_OVERFLOW    DISABLE_WARNING(-Wformat-overflow)
#else
  #define DISABLE_WARNING_PUSH
  #define DISABLE_WARNING_POP
  #define DISABLE_WARNING_PRINTF_FORMAT
  #define DISABLE_WARNING_PRINTF_FORMAT_EXTRA_ARGS
  #define DISABLE_WARNING_PRINTF_FORMAT_OVERFLOW 
#endif

#ifdef TEST_WITH_NON_STANDARD_FORMAT_STRINGS
DISABLE_WARNING_PUSH
DISABLE_WARNING_PRINTF_FORMAT
DISABLE_WARNING_PRINTF_FORMAT_EXTRA_ARGS
#endif

bool test_succeeded = true;

char* make_device_string(char const* s)
{
  size_t size = strlen(s) + 1;
  void* dsptr;
  cudaMalloc(&dsptr, size);
  cudaMemcpy(dsptr, s, size, cudaMemcpyDefault);
  cudaDeviceSynchronize();
  return (char *) dsptr;
}

inline char* mds(char const* s) { return make_device_string(s); }

struct poor_mans_string_view {
  char* data;
  size_t size;
};

struct sv_and_pos { 
	const poor_mans_string_view sv; 
	size_t pos; 
};

__device__ void append_to_buffer(char c, void* type_erased_svnp)
{ 
  auto& svnp = *(static_cast<sv_and_pos*>(type_erased_svnp));
  if (svnp.pos < svnp.sv.size) {
    svnp.sv.data[svnp.pos++] = c; 
  }
}

// output function type
typedef void (*out_fct_type)(char character, void* arg);

// ... just need to make the linker happy :-(
PRINTF_HOST void _putchar(char character)
{
  exit(EXIT_FAILURE);
}

enum class invokable {
  sprintf, vsprintf, snprintf, vsnprintf
};

__device__ __host__ char const* name(invokable inv) 
{
  switch(inv) {
  case invokable::sprintf:   return "sprintf";
  case invokable::snprintf:  return "snprintf";
  case invokable::vsprintf:  return "vsprintf";
  case invokable::vsnprintf: return "vsnprintf";
  }                     
  return "unknown";
}

__device__ int vsprintf_wrapper(char* buffer, char const* format, ...)
{
  va_list args;
  va_start(args, format);
  int ret = vsprintf_(buffer, format, args);
  va_end(args);
  return ret;
}

__device__ int vnsprintf_wrapper(char* buffer, size_t buffer_size, char const* format, ...)
{
  va_list args;
  va_start(args, format);
  int ret = vsnprintf_(buffer, buffer_size, format, args);
  va_end(args);
  return ret;
}

namespace kernels {

template <typename... Ts>
__global__ void 
invoke(
  int        * __restrict__  result, 
  invokable                  which, 
  char       * __restrict__  buffer, 
  size_t                     buffer_size, 
  char const * __restrict__  format, 
  Ts...                      args)
{
  switch(which) {
  case invokable::sprintf:   *result = sprintf_(buffer, format, args...); break;
  case invokable::snprintf:  *result = snprintf_(buffer, buffer_size, format, args...); break;
  case invokable::vsprintf:  *result = vsprintf_wrapper(buffer, format, args...); break;
  case invokable::vsnprintf: *result = vnsprintf_wrapper(buffer, buffer_size, format, args...); break;
  }
}

} // namespace kernels

template <typename... Ts>
int invoke_on_device(invokable which, char* buffer, size_t buffer_size, char const* format, Ts... args)
{
  char* buffer_d;
  char* format_d;
  int* result_d;
  int result;
  size_t format_size = strlen(format) + 1;
  cudaGetLastError(); // Clearing/ignoring earlier errors
  cudaMalloc(&result_d, sizeof(int));
  if (buffer != nullptr or buffer_size == 0) {
    cudaMalloc(&buffer_d, buffer_size);
    cudaMemcpy(buffer_d, buffer, buffer_size, cudaMemcpyDefault);
  } else {
	  buffer_d = nullptr;
  }
  if (format != nullptr or format_size == 0) {
    cudaMalloc(&format_d, format_size);
    cudaMemcpy(format_d, format, format_size, cudaMemcpyDefault);
  }
  else {
	  format_d = nullptr;
  }
  // std::cout << "Copying done, now launching kernel." << std::endl;
  kernels::invoke<<<1, 1>>>(result_d, which, buffer_d, buffer_size, format_d, args...); // Note: No perfect forwarding.
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) { 
    throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(error))); 
  }
  if (buffer != nullptr) {
    cudaMemcpy(buffer, buffer_d, buffer_size, cudaMemcpyDefault);
  }
  cudaMemcpy(&result, result_d, sizeof(int), cudaMemcpyDefault);
  cudaFree(buffer_d);
  cudaFree(format_d);
  cudaFree(result_d);
  cudaDeviceSynchronize();
  error = cudaGetLastError();
  if (error != cudaSuccess) { 
    throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(error))); 
  }
  return result;
}

constexpr const size_t base_buffer_size { 100 };


template <typename... Ts>
int invoke_on_device(invokable which, char* buffer, char const* format, Ts... args)
{
  return invoke_on_device<Ts...>(which, buffer, base_buffer_size, format, args...);
}

template <typename... Ts>
int printing_check(
  const char *expected,
  const char *,
  invokable invokable_printer,
  char *buffer,
  size_t buffer_size,
  const char *format,
  Ts &&... params)
{
  if (buffer == nullptr and expected != nullptr) {
    std::cerr << "Internal error: A null buffer is expected to become non-null" << std::endl;
    exit(EXIT_FAILURE);
  }
  auto ret = invoke_on_device(invokable_printer, buffer, buffer_size, format, std::forward<Ts>(params)...);
  // std::cout << "invoked_on_device with format \"" << format << "\" done." << std::endl;
  if (buffer == nullptr) {
    return ret;
  }
  if (buffer_size != base_buffer_size) {
    buffer[base_buffer_size - 1] = '\0';
  }
  //  std::cout << "----\n";
  //  std::cout << "Resulting buffer contents: " << '"' << buffer << '"' << '\n';
  if (strncmp(buffer, expected, buffer_size) != 0) {
    buffer[strlen(expected)] = '\0';
    std::cerr << "Failed with printer " << name(invokable_printer) <<
	    " with format \"" << format << "\":\n"
		<< "Actual:   \"" << buffer   << "\"\n"
		<< "Expected: \"" << expected << "\"\n" << std::flush;
    exit(EXIT_FAILURE);
  }
  return ret;
}

template <typename... Ts>
void printing_and_ret_check(
  int expected_return_value,
  const char *expected,
  const char *,
  invokable invokable_printer,
  char *buffer,
  size_t buffer_size,
  const char *format,
  Ts &&... params)
{
    auto ret = printing_check(expected, nullptr, invokable_printer, buffer, buffer_size, format, std::forward<Ts>(params)...);
    if (ret != expected_return_value) {
      std::cerr << "Unexpected return value with printer " << name(invokable_printer) <<
      " and format \"" << format << "\":\n    Actual: " << ret << "\n    Expected: " <<
      expected_return_value << std::endl;
      exit(EXIT_FAILURE);
    }
}

namespace kernels {

__global__ void fctprintf_kernel(char* buffer)
{
  sv_and_pos svnp { {buffer, base_buffer_size}, 0 };
  fctprintf(append_to_buffer, &svnp, "This is a test of %X", 0x12EFU);
}

} // namespace kernels

void testcase_fctprintf() {
  char buffer[base_buffer_size];
  char* buffer_d;
  cudaMalloc(&buffer_d, base_buffer_size);
  cudaMemset(buffer_d, 0xCC, base_buffer_size);
  kernels::fctprintf_kernel<<<1, 1>>>(buffer_d);
  cudaMemcpy(buffer, buffer_d, base_buffer_size, cudaMemcpyDefault);
  cudaDeviceSynchronize();
  if (strncmp(buffer, "This is a test of 12EF", 22U) != 0) {
    std::cerr << "fctprintf failed to produce the correct string." << std::endl;
    exit(EXIT_FAILURE);
  }
  // Remember: printf does not append a `\0` to the output after going through its format string.
  if (buffer[22] != (char)0xCC) {
    std::cerr << "fctprintf changed buffer characters past where it was allowed to\n" << std::endl;
    exit(EXIT_FAILURE);
  }
  cudaFree(buffer_d);
}

PRINTF_HD static void vfctprintfbuilder_1(out_fct_type f, void* f_arg, ...)
{
  va_list args;
  va_start(args, f_arg);
  vfctprintf(f, f_arg, "This is a test of %X", args);
  va_end(args);
}

namespace kernels {

__global__ void vfctprintf(char* buffer)
{
  sv_and_pos svnp { {buffer, base_buffer_size}, 0 };
  vfctprintfbuilder_1(append_to_buffer, &svnp, 0x12EFU);
}

} // namespace kernels

void testcase_vfctprintf() {
  char buffer[base_buffer_size];
  char* buffer_d;
  cudaMalloc(&buffer_d, base_buffer_size);
  cudaMemset(buffer_d, 0xCC, base_buffer_size);
  kernels::vfctprintf<<<1, 1>>>(buffer_d);
  cudaMemcpy(buffer, buffer_d, base_buffer_size, cudaMemcpyDefault);
  cudaDeviceSynchronize();
  if (strncmp(buffer, "This is a test of 12EF", 22U) != 0) {
    std::cerr << "vfctprintf failed to produce the correct string." << std::endl;
    exit(EXIT_FAILURE);
  }
  if (buffer[22] != (char)0xCC) {
    std::cerr << "vfctprintf changed buffer characters past where it was allowed to\n" << std::endl;
    exit(EXIT_FAILURE);
  }
  cudaFree(buffer_d);
}

//namespace kernels {
//
//__global__ void snprintf(char* buffer, size_t buffer_size)
//{
//  snprintf_(buffer, buffer_size, "%d", -1000);
//}
//
//} // namespace kernels

void testcase_snprintf() {
  char buffer[base_buffer_size];
  printing_check("-1000", "==", invokable::snprintf, buffer, base_buffer_size, "%d", -1000);
  printing_check("-1", "==", invokable::snprintf, buffer, 3, "%d", -1000);
}

void testcase_vsprintf() {
  char buffer[base_buffer_size];
  printing_check("-1", "==", invokable::vsprintf, buffer, base_buffer_size, "%d", -1 );
  printing_check("3 -1000 test", "==", invokable::vsprintf, buffer, base_buffer_size, "%d %d %s", 3, -1000, mds("test") );
}

void testcase_vsnprintf() {
  char buffer[base_buffer_size];
  printing_check("-1", "==", invokable::vsnprintf, buffer, base_buffer_size, "%d", -1);
  printing_check("3 -1000 test", "==", invokable::vsnprintf, buffer, base_buffer_size, "%d %d %s", 3, -1000, mds("test"));
}

void testcase_simple_sprintf() {
  char buffer[base_buffer_size];
  memset(buffer, 0xCC, base_buffer_size);
  printing_check("42", "==", invokable::sprintf, buffer, base_buffer_size, "%d", 42);
}

void testcase_space_flag() {
  char buffer[base_buffer_size];
  memset(buffer, 0xCC, base_buffer_size);
  printing_check(" 42", "==", invokable::sprintf, buffer, base_buffer_size, "% d", 42);
  printing_check("-42", "==", invokable::sprintf, buffer, base_buffer_size, "% d", -42);
  printing_check("   42", "==", invokable::sprintf, buffer, base_buffer_size, "% 5d", 42);
  printing_check("  -42", "==", invokable::sprintf, buffer, base_buffer_size, "% 5d", -42);
  printing_check("             42", "==", invokable::sprintf, buffer, base_buffer_size, "% 15d", 42);
  printing_check("            -42", "==", invokable::sprintf, buffer, base_buffer_size, "% 15d", -42);
  printing_check("            -42", "==", invokable::sprintf, buffer, base_buffer_size, "% 15d", -42);
  printing_check("        -42.987", "==", invokable::sprintf, buffer, base_buffer_size, "% 15.3f", -42.987);
  printing_check("         42.987", "==", invokable::sprintf, buffer, base_buffer_size, "% 15.3f", 42.987);
  printing_check(" 1024", "==", invokable::sprintf, buffer, base_buffer_size, "% d", 1024);
  printing_check("-1024", "==", invokable::sprintf, buffer, base_buffer_size, "% d", -1024);
  printing_check(" 1024", "==", invokable::sprintf, buffer, base_buffer_size, "% i", 1024);
  printing_check("-1024", "==", invokable::sprintf, buffer, base_buffer_size, "% i", -1024);
}


#ifdef TEST_WITH_NON_STANDARD_FORMAT_STRINGS
void testcase_space_flag__non_standard_format() {
  char buffer[base_buffer_size];
  printing_check("Hello testing", "==", invokable::sprintf, buffer, base_buffer_size, "% s", mds("Hello testing"));
  printing_check("1024", "==", invokable::sprintf, buffer, base_buffer_size, "% u", 1024);
  printing_check("4294966272", "==", invokable::sprintf, buffer, base_buffer_size, "% u", 4294966272U);
  printing_check("777", "==", invokable::sprintf, buffer, base_buffer_size, "% o", 511);
  printing_check("37777777001", "==", invokable::sprintf, buffer, base_buffer_size, "% o", 4294966785U);
  printing_check("1234abcd", "==", invokable::sprintf, buffer, base_buffer_size, "% x", 305441741);
  printing_check("edcb5433", "==", invokable::sprintf, buffer, base_buffer_size, "% x", 3989525555U);
  printing_check("1234ABCD", "==", invokable::sprintf, buffer, base_buffer_size, "% X", 305441741);
  printing_check("EDCB5433", "==", invokable::sprintf, buffer, base_buffer_size, "% X", 3989525555U);
  printing_check("x", "==", invokable::sprintf, buffer, base_buffer_size, "% c", 'x');
}
#endif

void testcase_plus_flag() {
  char buffer[base_buffer_size];
  printing_check("+42", "==", invokable::sprintf, buffer, base_buffer_size, "%+d", 42);
  printing_check("-42", "==", invokable::sprintf, buffer, base_buffer_size, "%+d", -42);
  printing_check("  +42", "==", invokable::sprintf, buffer, base_buffer_size, "%+5d", 42);
  printing_check("  -42", "==", invokable::sprintf, buffer, base_buffer_size, "%+5d", -42);
  printing_check("            +42", "==", invokable::sprintf, buffer, base_buffer_size, "%+15d", 42);
  printing_check("            -42", "==", invokable::sprintf, buffer, base_buffer_size, "%+15d", -42);
  printing_check("+1024", "==", invokable::sprintf, buffer, base_buffer_size, "%+d", 1024);
  printing_check("-1024", "==", invokable::sprintf, buffer, base_buffer_size, "%+d", -1024);
  printing_check("+1024", "==", invokable::sprintf, buffer, base_buffer_size, "%+i", 1024);
  printing_check("-1024", "==", invokable::sprintf, buffer, base_buffer_size, "%+i", -1024);
  printing_check("+", "==", invokable::sprintf, buffer, base_buffer_size, "%+.0d", 0);
}

#ifdef TEST_WITH_NON_STANDARD_FORMAT_STRINGS
void testcase_plus_flag__non_standard_format() {
  char buffer[base_buffer_size];
  printing_check("Hello testing", "==", invokable::sprintf, buffer, base_buffer_size, "%+s", mds("Hello testing"));
  printing_check("1024", "==", invokable::sprintf, buffer, base_buffer_size, "%+u", 1024);
  printing_check("4294966272", "==", invokable::sprintf, buffer, base_buffer_size, "%+u", 4294966272U);
  printing_check("777", "==", invokable::sprintf, buffer, base_buffer_size, "%+o", 511);
  printing_check("37777777001", "==", invokable::sprintf, buffer, base_buffer_size, "%+o", 4294966785U);
  printing_check("1234abcd", "==", invokable::sprintf, buffer, base_buffer_size, "%+x", 305441741);
  printing_check("edcb5433", "==", invokable::sprintf, buffer, base_buffer_size, "%+x", 3989525555U);
  printing_check("1234ABCD", "==", invokable::sprintf, buffer, base_buffer_size, "%+X", 305441741);
  printing_check("EDCB5433", "==", invokable::sprintf, buffer, base_buffer_size, "%+X", 3989525555U);
  printing_check("x", "==", invokable::sprintf, buffer, base_buffer_size, "%+c", 'x');
}
#endif


void testcase_0_flag() {
  char buffer[base_buffer_size];
  printing_check("42", "==", invokable::sprintf, buffer, base_buffer_size, "%0d", 42);
  printing_check("42", "==", invokable::sprintf, buffer, base_buffer_size, "%0ld", 42L);
  printing_check("-42", "==", invokable::sprintf, buffer, base_buffer_size, "%0d", -42);
  printing_check("00042", "==", invokable::sprintf, buffer, base_buffer_size, "%05d", 42);
  printing_check("-0042", "==", invokable::sprintf, buffer, base_buffer_size, "%05d", -42);
  printing_check("000000000000042", "==", invokable::sprintf, buffer, base_buffer_size, "%015d", 42);
  printing_check("-00000000000042", "==", invokable::sprintf, buffer, base_buffer_size, "%015d", -42);
  printing_check("000000000042.12", "==", invokable::sprintf, buffer, base_buffer_size, "%015.2f", 42.1234);
  printing_check("00000000042.988", "==", invokable::sprintf, buffer, base_buffer_size, "%015.3f", 42.9876);
  printing_check("-00000042.98760", "==", invokable::sprintf, buffer, base_buffer_size, "%015.5f", -42.9876);
}


void testcase_minus_flag() {
  char buffer[base_buffer_size];
  printing_check("42", "==", invokable::sprintf, buffer, base_buffer_size, "%-d", 42);
  printing_check("-42", "==", invokable::sprintf, buffer, base_buffer_size, "%-d", -42);
  printing_check("42   ", "==", invokable::sprintf, buffer, base_buffer_size, "%-5d", 42);
  printing_check("-42  ", "==", invokable::sprintf, buffer, base_buffer_size, "%-5d", -42);
  printing_check("42             ", "==", invokable::sprintf, buffer, base_buffer_size, "%-15d", 42);
  printing_check("-42            ", "==", invokable::sprintf, buffer, base_buffer_size, "%-15d", -42);
}


#ifdef TEST_WITH_NON_STANDARD_FORMAT_STRINGS
void testcase_minus_flag__non_standard_format() {
  char buffer[base_buffer_size];
  printing_check("42", "==", invokable::sprintf, buffer, base_buffer_size, "%-0d", 42);
  printing_check("-42", "==", invokable::sprintf, buffer, base_buffer_size, "%-0d", -42);
  printing_check("42   ", "==", invokable::sprintf, buffer, base_buffer_size, "%-05d", 42);
  printing_check("-42  ", "==", invokable::sprintf, buffer, base_buffer_size, "%-05d", -42);
  printing_check("42             ", "==", invokable::sprintf, buffer, base_buffer_size, "%-015d", 42);
  printing_check("-42            ", "==", invokable::sprintf, buffer, base_buffer_size, "%-015d", -42);
  printing_check("42", "==", invokable::sprintf, buffer, base_buffer_size, "%0-d", 42);
  printing_check("-42", "==", invokable::sprintf, buffer, base_buffer_size, "%0-d", -42);
  printing_check("42   ", "==", invokable::sprintf, buffer, base_buffer_size, "%0-5d", 42);
  printing_check("-42  ", "==", invokable::sprintf, buffer, base_buffer_size, "%0-5d", -42);
  printing_check("42             ", "==", invokable::sprintf, buffer, base_buffer_size, "%0-15d", 42);
  printing_check("-42            ", "==", invokable::sprintf, buffer, base_buffer_size, "%0-15d", -42);

#ifndef PRINTF_DISABLE_SUPPORT_EXPONENTIAL
  printing_check("-4.200e+01     ", "==", invokable::sprintf, buffer, base_buffer_size, "%0-15.3e", -42.);
#else
  printing_check("e", "==", invokable::sprintf, buffer, base_buffer_size, "%0-15.3e", -42.);
#endif

#ifndef PRINTF_DISABLE_SUPPORT_EXPONENTIAL
  printing_check("-42            ", "==", invokable::sprintf, buffer, base_buffer_size, "%0-15.3g", -42.);
#else
  printing_check("g", "==", invokable::sprintf, buffer, base_buffer_size, "%0-15.3g", -42.);
#endif
}
#endif


void testcase_hash_flag() {
  char buffer[base_buffer_size];
  printing_check("", "==", invokable::sprintf, buffer, base_buffer_size, "%#.0x", 0);
  printing_check("0", "==", invokable::sprintf, buffer, base_buffer_size, "%#.1x", 0);
  printing_check("", "==", invokable::sprintf, buffer, base_buffer_size, "%#.0llx", (long long)0);
  printing_check("0x0000614e", "==", invokable::sprintf, buffer, base_buffer_size, "%#.8x", 0x614e);
}

#ifdef TEST_WITH_NON_STANDARD_FORMAT_STRINGS
void testcase_hash_flag__non_standard_format() {
  char buffer[base_buffer_size ];
  printing_check("0b110", "==", invokable::sprintf, buffer, base_buffer_size, "%#b", 6);
}
#endif

void testcase_specifier() {
  char buffer[base_buffer_size];

  printing_check("Hello testing", "==", invokable::sprintf, buffer, base_buffer_size, "Hello testing");
  printing_check("Hello testing", "==", invokable::sprintf, buffer, base_buffer_size, "%s", mds("Hello testing"));

DISABLE_WARNING_PUSH
DISABLE_WARNING_PRINTF_FORMAT_OVERFLOW
  printing_check("(null)", "==", invokable::sprintf, buffer, base_buffer_size, "%s", (const char*) NULL);
DISABLE_WARNING_POP
  printing_check("1024", "==", invokable::sprintf, buffer, base_buffer_size, "%d", 1024);
  printing_check("-1024", "==", invokable::sprintf, buffer, base_buffer_size, "%d", -1024);
  printing_check("1024", "==", invokable::sprintf, buffer, base_buffer_size, "%i", 1024);
  printing_check("-1024", "==", invokable::sprintf, buffer, base_buffer_size, "%i", -1024);
  printing_check("1024", "==", invokable::sprintf, buffer, base_buffer_size, "%u", 1024);
  printing_check("4294966272", "==", invokable::sprintf, buffer, base_buffer_size, "%u", 4294966272U);
  printing_check("777", "==", invokable::sprintf, buffer, base_buffer_size, "%o", 511);
  printing_check("37777777001", "==", invokable::sprintf, buffer, base_buffer_size, "%o", 4294966785U);
  printing_check("1234abcd", "==", invokable::sprintf, buffer, base_buffer_size, "%x", 305441741);
  printing_check("edcb5433", "==", invokable::sprintf, buffer, base_buffer_size, "%x", 3989525555U);
  printing_check("1234ABCD", "==", invokable::sprintf, buffer, base_buffer_size, "%X", 305441741);
  printing_check("EDCB5433", "==", invokable::sprintf, buffer, base_buffer_size, "%X", 3989525555U);
  printing_check("%", "==", invokable::sprintf, buffer, base_buffer_size, "%%");
}


void testcase_width() {
  char buffer[base_buffer_size];
  printing_check("Hello testing", "==", invokable::sprintf, buffer, base_buffer_size, "%1s", mds("Hello testing"));
  printing_check("1024", "==", invokable::sprintf, buffer, base_buffer_size, "%1d", 1024);
  printing_check("-1024", "==", invokable::sprintf, buffer, base_buffer_size, "%1d", -1024);
  printing_check("1024", "==", invokable::sprintf, buffer, base_buffer_size, "%1i", 1024);
  printing_check("-1024", "==", invokable::sprintf, buffer, base_buffer_size, "%1i", -1024);
  printing_check("1024", "==", invokable::sprintf, buffer, base_buffer_size, "%1u", 1024);
  printing_check("4294966272", "==", invokable::sprintf, buffer, base_buffer_size, "%1u", 4294966272U);
  printing_check("777", "==", invokable::sprintf, buffer, base_buffer_size, "%1o", 511);
  printing_check("37777777001", "==", invokable::sprintf, buffer, base_buffer_size, "%1o", 4294966785U);
  printing_check("1234abcd", "==", invokable::sprintf, buffer, base_buffer_size, "%1x", 305441741);
  printing_check("edcb5433", "==", invokable::sprintf, buffer, base_buffer_size, "%1x", 3989525555U);
  printing_check("1234ABCD", "==", invokable::sprintf, buffer, base_buffer_size, "%1X", 305441741);
  printing_check("EDCB5433", "==", invokable::sprintf, buffer, base_buffer_size, "%1X", 3989525555U);
  printing_check("x", "==", invokable::sprintf, buffer, base_buffer_size, "%1c", 'x');
}


void testcase_width_20() {
  char buffer[base_buffer_size];
  printing_check("               Hello", "==", invokable::sprintf, buffer, base_buffer_size, "%20s", mds("Hello"));
  printing_check("                1024", "==", invokable::sprintf, buffer, base_buffer_size, "%20d", 1024);
  printing_check("               -1024", "==", invokable::sprintf, buffer, base_buffer_size, "%20d", -1024);
  printing_check("                1024", "==", invokable::sprintf, buffer, base_buffer_size, "%20i", 1024);
  printing_check("               -1024", "==", invokable::sprintf, buffer, base_buffer_size, "%20i", -1024);
  printing_check("                1024", "==", invokable::sprintf, buffer, base_buffer_size, "%20u", 1024);
  printing_check("          4294966272", "==", invokable::sprintf, buffer, base_buffer_size, "%20u", 4294966272U);
  printing_check("                 777", "==", invokable::sprintf, buffer, base_buffer_size, "%20o", 511);
  printing_check("         37777777001", "==", invokable::sprintf, buffer, base_buffer_size, "%20o", 4294966785U);
  printing_check("            1234abcd", "==", invokable::sprintf, buffer, base_buffer_size, "%20x", 305441741);
  printing_check("            edcb5433", "==", invokable::sprintf, buffer, base_buffer_size, "%20x", 3989525555U);
  printing_check("            1234ABCD", "==", invokable::sprintf, buffer, base_buffer_size, "%20X", 305441741);
  printing_check("            EDCB5433", "==", invokable::sprintf, buffer, base_buffer_size, "%20X", 3989525555U);
  printing_check("                   x", "==", invokable::sprintf, buffer, base_buffer_size, "%20c", 'x');
}


void testcase_width_star_20() {
  char buffer[base_buffer_size];
  printing_check("               Hello", "==", invokable::sprintf, buffer, base_buffer_size, "%*s", 20, mds("Hello"));
  printing_check("                1024", "==", invokable::sprintf, buffer, base_buffer_size, "%*d", 20, 1024);
  printing_check("               -1024", "==", invokable::sprintf, buffer, base_buffer_size, "%*d", 20, -1024);
  printing_check("                1024", "==", invokable::sprintf, buffer, base_buffer_size, "%*i", 20, 1024);
  printing_check("               -1024", "==", invokable::sprintf, buffer, base_buffer_size, "%*i", 20, -1024);
  printing_check("                1024", "==", invokable::sprintf, buffer, base_buffer_size, "%*u", 20, 1024);
  printing_check("          4294966272", "==", invokable::sprintf, buffer, base_buffer_size, "%*u", 20, 4294966272U);
  printing_check("                 777", "==", invokable::sprintf, buffer, base_buffer_size, "%*o", 20, 511);
  printing_check("         37777777001", "==", invokable::sprintf, buffer, base_buffer_size, "%*o", 20, 4294966785U);
  printing_check("            1234abcd", "==", invokable::sprintf, buffer, base_buffer_size, "%*x", 20, 305441741);
  printing_check("            edcb5433", "==", invokable::sprintf, buffer, base_buffer_size, "%*x", 20, 3989525555U);
  printing_check("            1234ABCD", "==", invokable::sprintf, buffer, base_buffer_size, "%*X", 20, 305441741);
  printing_check("            EDCB5433", "==", invokable::sprintf, buffer, base_buffer_size, "%*X", 20, 3989525555U);
  printing_check("                   x", "==", invokable::sprintf, buffer, base_buffer_size, "%*c", 20,'x');
}


void testcase_width_minus_20() {
  char buffer[base_buffer_size];
  printing_check("Hello               ", "==", invokable::sprintf, buffer, base_buffer_size, "%-20s", mds("Hello"));
  printing_check("1024                ", "==", invokable::sprintf, buffer, base_buffer_size, "%-20d", 1024);
  printing_check("-1024               ", "==", invokable::sprintf, buffer, base_buffer_size, "%-20d", -1024);
  printing_check("1024                ", "==", invokable::sprintf, buffer, base_buffer_size, "%-20i", 1024);
  printing_check("-1024               ", "==", invokable::sprintf, buffer, base_buffer_size, "%-20i", -1024);
  printing_check("1024                ", "==", invokable::sprintf, buffer, base_buffer_size, "%-20u", 1024);
  printing_check("1024.1234           ", "==", invokable::sprintf, buffer, base_buffer_size, "%-20.4f", 1024.1234);
  printing_check("4294966272          ", "==", invokable::sprintf, buffer, base_buffer_size, "%-20u", 4294966272U);
  printing_check("777                 ", "==", invokable::sprintf, buffer, base_buffer_size, "%-20o", 511);
  printing_check("37777777001         ", "==", invokable::sprintf, buffer, base_buffer_size, "%-20o", 4294966785U);
  printing_check("1234abcd            ", "==", invokable::sprintf, buffer, base_buffer_size, "%-20x", 305441741);
  printing_check("edcb5433            ", "==", invokable::sprintf, buffer, base_buffer_size, "%-20x", 3989525555U);
  printing_check("1234ABCD            ", "==", invokable::sprintf, buffer, base_buffer_size, "%-20X", 305441741);
  printing_check("EDCB5433            ", "==", invokable::sprintf, buffer, base_buffer_size, "%-20X", 3989525555U);
  printing_check("x                   ", "==", invokable::sprintf, buffer, base_buffer_size, "%-20c", 'x');
  printing_check("|    9| |9 | |    9|", "==", invokable::sprintf, buffer, base_buffer_size, "|%5d| |%-2d| |%5d|", 9, 9, 9);
  printing_check("|   10| |10| |   10|", "==", invokable::sprintf, buffer, base_buffer_size, "|%5d| |%-2d| |%5d|", 10, 10, 10);
  printing_check("|    9| |9           | |    9|", "==", invokable::sprintf, buffer, base_buffer_size, "|%5d| |%-12d| |%5d|", 9, 9, 9);
  printing_check("|   10| |10          | |   10|", "==", invokable::sprintf, buffer, base_buffer_size, "|%5d| |%-12d| |%5d|", 10, 10, 10);
}

#ifdef TEST_WITH_NON_STANDARD_FORMAT_STRINGS
void testcase_width_0_minus_20() {
  char buffer[base_buffer_size];
  printing_check("Hello               ", "==", invokable::sprintf, buffer, base_buffer_size, "%0-20s", mds("Hello"));
  printing_check("1024                ", "==", invokable::sprintf, buffer, base_buffer_size, "%0-20d", 1024);
  printing_check("-1024               ", "==", invokable::sprintf, buffer, base_buffer_size, "%0-20d", -1024);
  printing_check("1024                ", "==", invokable::sprintf, buffer, base_buffer_size, "%0-20i", 1024);
  printing_check("-1024               ", "==", invokable::sprintf, buffer, base_buffer_size, "%0-20i", -1024);
  printing_check("1024                ", "==", invokable::sprintf, buffer, base_buffer_size, "%0-20u", 1024);
  printing_check("4294966272          ", "==", invokable::sprintf, buffer, base_buffer_size, "%0-20u", 4294966272U);
  printing_check("777                 ", "==", invokable::sprintf, buffer, base_buffer_size, "%0-20o", 511);
  printing_check("37777777001         ", "==", invokable::sprintf, buffer, base_buffer_size, "%0-20o", 4294966785U);
  printing_check("1234abcd            ", "==", invokable::sprintf, buffer, base_buffer_size, "%0-20x", 305441741);
  printing_check("edcb5433            ", "==", invokable::sprintf, buffer, base_buffer_size, "%0-20x", 3989525555U);
  printing_check("1234ABCD            ", "==", invokable::sprintf, buffer, base_buffer_size, "%0-20X", 305441741);
  printing_check("EDCB5433            ", "==", invokable::sprintf, buffer, base_buffer_size, "%0-20X", 3989525555U);
  printing_check("x                   ", "==", invokable::sprintf, buffer, base_buffer_size, "%0-20c", 'x');
}
#endif

void testcase_padding_20() {
  char buffer[base_buffer_size];
  printing_check("00000000000000001024", "==", invokable::sprintf, buffer, base_buffer_size, "%020d", 1024);
  printing_check("-0000000000000001024", "==", invokable::sprintf, buffer, base_buffer_size, "%020d", -1024);
  printing_check("00000000000000001024", "==", invokable::sprintf, buffer, base_buffer_size, "%020i", 1024);
  printing_check("-0000000000000001024", "==", invokable::sprintf, buffer, base_buffer_size, "%020i", -1024);
  printing_check("00000000000000001024", "==", invokable::sprintf, buffer, base_buffer_size, "%020u", 1024);
  printing_check("00000000004294966272", "==", invokable::sprintf, buffer, base_buffer_size, "%020u", 4294966272U);
  printing_check("00000000000000000777", "==", invokable::sprintf, buffer, base_buffer_size, "%020o", 511);
  printing_check("00000000037777777001", "==", invokable::sprintf, buffer, base_buffer_size, "%020o", 4294966785U);
  printing_check("0000000000001234abcd", "==", invokable::sprintf, buffer, base_buffer_size, "%020x", 305441741);
  printing_check("000000000000edcb5433", "==", invokable::sprintf, buffer, base_buffer_size, "%020x", 3989525555U);
  printing_check("0000000000001234ABCD", "==", invokable::sprintf, buffer, base_buffer_size, "%020X", 305441741);
  printing_check("000000000000EDCB5433", "==", invokable::sprintf, buffer, base_buffer_size, "%020X", 3989525555U);
}


void testcase_padding_dot_20() {
  char buffer[base_buffer_size];
  printing_check("00000000000000001024", "==", invokable::sprintf, buffer, base_buffer_size, "%.20d", 1024);
  printing_check("-00000000000000001024", "==", invokable::sprintf, buffer, base_buffer_size, "%.20d", -1024);
  printing_check("00000000000000001024", "==", invokable::sprintf, buffer, base_buffer_size, "%.20i", 1024);
  printing_check("-00000000000000001024", "==", invokable::sprintf, buffer, base_buffer_size, "%.20i", -1024);
  printing_check("00000000000000001024", "==", invokable::sprintf, buffer, base_buffer_size, "%.20u", 1024);
  printing_check("00000000004294966272", "==", invokable::sprintf, buffer, base_buffer_size, "%.20u", 4294966272U);
  printing_check("00000000000000000777", "==", invokable::sprintf, buffer, base_buffer_size, "%.20o", 511);
  printing_check("00000000037777777001", "==", invokable::sprintf, buffer, base_buffer_size, "%.20o", 4294966785U);
  printing_check("0000000000001234abcd", "==", invokable::sprintf, buffer, base_buffer_size, "%.20x", 305441741);
  printing_check("000000000000edcb5433", "==", invokable::sprintf, buffer, base_buffer_size, "%.20x", 3989525555U);
  printing_check("0000000000001234ABCD", "==", invokable::sprintf, buffer, base_buffer_size, "%.20X", 305441741);
  printing_check("000000000000EDCB5433", "==", invokable::sprintf, buffer, base_buffer_size, "%.20X", 3989525555U);
}

#ifdef TEST_WITH_NON_STANDARD_FORMAT_STRINGS
void testcase_padding_hash_020__non_standard_format() {
  char buffer[base_buffer_size];
  printing_check("00000000000000001024", "==", invokable::sprintf, buffer, base_buffer_size, "%#020d", 1024);
  printing_check("-0000000000000001024", "==", invokable::sprintf, buffer, base_buffer_size, "%#020d", -1024);
  printing_check("00000000000000001024", "==", invokable::sprintf, buffer, base_buffer_size, "%#020i", 1024);
  printing_check("-0000000000000001024", "==", invokable::sprintf, buffer, base_buffer_size, "%#020i", -1024);
  printing_check("00000000000000001024", "==", invokable::sprintf, buffer, base_buffer_size, "%#020u", 1024);
  printing_check("00000000004294966272", "==", invokable::sprintf, buffer, base_buffer_size, "%#020u", 4294966272U);
}
#endif

void testcase_padding_hash_020() {
  char buffer[base_buffer_size];
  printing_check("00000000000000000777", "==", invokable::sprintf, buffer, base_buffer_size, "%#020o", 511);
  printing_check("00000000037777777001", "==", invokable::sprintf, buffer, base_buffer_size, "%#020o", 4294966785U);
  printing_check("0x00000000001234abcd", "==", invokable::sprintf, buffer, base_buffer_size, "%#020x", 305441741);
  printing_check("0x0000000000edcb5433", "==", invokable::sprintf, buffer, base_buffer_size, "%#020x", 3989525555U);
  printing_check("0X00000000001234ABCD", "==", invokable::sprintf, buffer, base_buffer_size, "%#020X", 305441741);
  printing_check("0X0000000000EDCB5433", "==", invokable::sprintf, buffer, base_buffer_size, "%#020X", 3989525555U);
}


#ifdef TEST_WITH_NON_STANDARD_FORMAT_STRINGS
void testcase_padding_hash_20__non_standard_format() {
  char buffer[base_buffer_size];
  printing_check("                1024", "==", invokable::sprintf, buffer, base_buffer_size, "%#20d", 1024);
  printing_check("               -1024", "==", invokable::sprintf, buffer, base_buffer_size, "%#20d", -1024);
  printing_check("                1024", "==", invokable::sprintf, buffer, base_buffer_size, "%#20i", 1024);
  printing_check("               -1024", "==", invokable::sprintf, buffer, base_buffer_size, "%#20i", -1024);
  printing_check("                1024", "==", invokable::sprintf, buffer, base_buffer_size, "%#20u", 1024);
  printing_check("          4294966272", "==", invokable::sprintf, buffer, base_buffer_size, "%#20u", 4294966272U);
}
#endif

void testcase_padding_hash_20() {
  char buffer[base_buffer_size];
  printing_check("                0777", "==", invokable::sprintf, buffer, base_buffer_size, "%#20o", 511);
  printing_check("        037777777001", "==", invokable::sprintf, buffer, base_buffer_size, "%#20o", 4294966785U);
  printing_check("          0x1234abcd", "==", invokable::sprintf, buffer, base_buffer_size, "%#20x", 305441741);
  printing_check("          0xedcb5433", "==", invokable::sprintf, buffer, base_buffer_size, "%#20x", 3989525555U);
  printing_check("          0X1234ABCD", "==", invokable::sprintf, buffer, base_buffer_size, "%#20X", 305441741);
  printing_check("          0XEDCB5433", "==", invokable::sprintf, buffer, base_buffer_size, "%#20X", 3989525555U);
}


void testcase_padding_20_dot_5() {
  char buffer[base_buffer_size];
  printing_check("               01024", "==", invokable::sprintf, buffer, base_buffer_size, "%20.5d", 1024);
  printing_check("              -01024", "==", invokable::sprintf, buffer, base_buffer_size, "%20.5d", -1024);
  printing_check("               01024", "==", invokable::sprintf, buffer, base_buffer_size, "%20.5i", 1024);
  printing_check("              -01024", "==", invokable::sprintf, buffer, base_buffer_size, "%20.5i", -1024);
  printing_check("               01024", "==", invokable::sprintf, buffer, base_buffer_size, "%20.5u", 1024);
  printing_check("          4294966272", "==", invokable::sprintf, buffer, base_buffer_size, "%20.5u", 4294966272U);
  printing_check("               00777", "==", invokable::sprintf, buffer, base_buffer_size, "%20.5o", 511);
  printing_check("         37777777001", "==", invokable::sprintf, buffer, base_buffer_size, "%20.5o", 4294966785U);
  printing_check("            1234abcd", "==", invokable::sprintf, buffer, base_buffer_size, "%20.5x", 305441741);
  printing_check("          00edcb5433", "==", invokable::sprintf, buffer, base_buffer_size, "%20.10x", 3989525555U);
  printing_check("            1234ABCD", "==", invokable::sprintf, buffer, base_buffer_size, "%20.5X", 305441741);
  printing_check("          00EDCB5433", "==", invokable::sprintf, buffer, base_buffer_size, "%20.10X", 3989525555U);
}


void testcase_padding_neg_numbers() {
  char buffer[base_buffer_size];

  // space padding
  printing_check("-5", "==", invokable::sprintf, buffer, base_buffer_size, "% 1d", -5);
  printing_check("-5", "==", invokable::sprintf, buffer, base_buffer_size, "% 2d", -5);
  printing_check(" -5", "==", invokable::sprintf, buffer, base_buffer_size, "% 3d", -5);
  printing_check("  -5", "==", invokable::sprintf, buffer, base_buffer_size, "% 4d", -5);

  // zero padding
  printing_check("-5", "==", invokable::sprintf, buffer, base_buffer_size, "%01d", -5);
  printing_check("-5", "==", invokable::sprintf, buffer, base_buffer_size, "%02d", -5);
  printing_check("-05", "==", invokable::sprintf, buffer, base_buffer_size, "%03d", -5);
  printing_check("-005", "==", invokable::sprintf, buffer, base_buffer_size, "%04d", -5);
}


void testcase_float_padding_neg_numbers() {
  char buffer[base_buffer_size];
/*
  // space padding
  printing_check("-5.0", "==", invokable::sprintf, buffer, base_buffer_size, "% 3.1f", -5.);
  printing_check("-5.0", "==", invokable::sprintf, buffer, base_buffer_size, "% 4.1f", -5.);
  printing_check(" -5.0", "==", invokable::sprintf, buffer, base_buffer_size, "% 5.1f", -5.);
*/
#ifndef PRINTF_DISABLE_SUPPORT_EXPONENTIAL
  printing_check("    -5", "==", invokable::sprintf, buffer, base_buffer_size, "% 6.1g", -5.);
/*  printing_check("-5.0e+00", "==", invokable::sprintf, buffer, base_buffer_size, "% 6.1e", -5.);
  printing_check("  -5.0e+00", "==", invokable::sprintf, buffer, base_buffer_size, "% 10.1e", -5.);
*/
#endif
/*
  // zero padding
  printing_check("-5.0", "==", invokable::sprintf, buffer, base_buffer_size, "%03.1f", -5.);
  printing_check("-5.0", "==", invokable::sprintf, buffer, base_buffer_size, "%04.1f", -5.);
  printing_check("-05.0", "==", invokable::sprintf, buffer, base_buffer_size, "%05.1f", -5.);

  // zero padding no decimal point
  printing_check("-5", "==", invokable::sprintf, buffer, base_buffer_size, "%01.0f", -5.);
  printing_check("-5", "==", invokable::sprintf, buffer, base_buffer_size, "%02.0f", -5.);
  printing_check("-05", "==", invokable::sprintf, buffer, base_buffer_size, "%03.0f", -5.);

#ifndef PRINTF_DISABLE_SUPPORT_EXPONENTIAL
  printing_check("-005.0e+00", "==", invokable::sprintf, buffer, base_buffer_size, "%010.1e", -5.);
  printing_check("-05E+00", "==", invokable::sprintf, buffer, base_buffer_size, "%07.0E", -5.);
  printing_check("-05", "==", invokable::sprintf, buffer, base_buffer_size, "%03.0g", -5.);
#endif
 */
}

void testcase_length() {
  char buffer[base_buffer_size];
  printing_check("", "==", invokable::sprintf, buffer, base_buffer_size, "%.0s", mds("Hello testing"));
  printing_check("                    ", "==", invokable::sprintf, buffer, base_buffer_size, "%20.0s", mds("Hello testing"));
  printing_check("", "==", invokable::sprintf, buffer, base_buffer_size, "%.s", mds("Hello testing"));
  printing_check("                    ", "==", invokable::sprintf, buffer, base_buffer_size, "%20.s", mds("Hello testing"));
  printing_check("                1024", "==", invokable::sprintf, buffer, base_buffer_size, "%20.0d", 1024);
  printing_check("               -1024", "==", invokable::sprintf, buffer, base_buffer_size, "%20.0d", -1024);
  printing_check("                    ", "==", invokable::sprintf, buffer, base_buffer_size, "%20.d", 0);
  printing_check("                1024", "==", invokable::sprintf, buffer, base_buffer_size, "%20.0i", 1024);
  printing_check("               -1024", "==", invokable::sprintf, buffer, base_buffer_size, "%20.i", -1024);
  printing_check("                    ", "==", invokable::sprintf, buffer, base_buffer_size, "%20.i", 0);
  printing_check("                1024", "==", invokable::sprintf, buffer, base_buffer_size, "%20.u", 1024);
  printing_check("          4294966272", "==", invokable::sprintf, buffer, base_buffer_size, "%20.0u", 4294966272U);
  printing_check("                    ", "==", invokable::sprintf, buffer, base_buffer_size, "%20.u", 0U);
  printing_check("                 777", "==", invokable::sprintf, buffer, base_buffer_size, "%20.o", 511);
  printing_check("         37777777001", "==", invokable::sprintf, buffer, base_buffer_size, "%20.0o", 4294966785U);
  printing_check("                    ", "==", invokable::sprintf, buffer, base_buffer_size, "%20.o", 0U);
  printing_check("            1234abcd", "==", invokable::sprintf, buffer, base_buffer_size, "%20.x", 305441741);
  printing_check("                                          1234abcd", "==", invokable::sprintf, buffer, base_buffer_size, "%50.x", 305441741);
  printing_check("                                          1234abcd     12345", "==", invokable::sprintf, buffer, base_buffer_size, "%50.x%10.u", 305441741, 12345);
  printing_check("            edcb5433", "==", invokable::sprintf, buffer, base_buffer_size, "%20.0x", 3989525555U);
  printing_check("                    ", "==", invokable::sprintf, buffer, base_buffer_size, "%20.x", 0U);
  printing_check("            1234ABCD", "==", invokable::sprintf, buffer, base_buffer_size, "%20.X", 305441741);
  printing_check("            EDCB5433", "==", invokable::sprintf, buffer, base_buffer_size, "%20.0X", 3989525555U);
  printing_check("                    ", "==", invokable::sprintf, buffer, base_buffer_size, "%20.X", 0U);
}

#ifdef TEST_WITH_NON_STANDARD_FORMAT_STRINGS
void testcase_length__non_standard_format() {
  char buffer[base_buffer_size];
  printing_check("  ", "==", invokable::sprintf, buffer, base_buffer_size, "%02.0u", 0U);
  printing_check("  ", "==", invokable::sprintf, buffer, base_buffer_size, "%02.0d", 0);
}
#endif


void testcase_float() {
  char buffer[base_buffer_size];

  // test special-case floats using math.h macros
  printing_check("     nan", "==", invokable::sprintf, buffer, base_buffer_size, "%8f", (double) NAN);
  printing_check("     inf", "==", invokable::sprintf, buffer, base_buffer_size, "%8f", (double) INFINITY);
  printing_check("-inf    ", "==", invokable::sprintf, buffer, base_buffer_size, "%-8f", (double) -INFINITY);

#ifndef PRINTF_DISABLE_SUPPORT_EXPONENTIAL
  printing_check("    +inf", "==", invokable::sprintf, buffer, base_buffer_size, "%+8e", (double) INFINITY);
#endif
  printing_check("3.1415", "==", invokable::sprintf, buffer, base_buffer_size, "%.4f", 3.1415354);
  printing_check("30343.142", "==", invokable::sprintf, buffer, base_buffer_size, "%.3f", 30343.1415354);

  // switch from decimal to exponential representation
  //
  if (PRINTF_MAX_INTEGRAL_DIGITS_FOR_DECIMAL < 3) {
    printing_check("1e+3", "==", invokable::sprintf, buffer, base_buffer_size, "%.0f", (double) ((int64_t)1 * 1000 ) );
  }
  else {
    printing_check("1000", "==", invokable::sprintf, buffer, base_buffer_size, "%.0f", (double) ((int64_t)1 * 1000 ) );
  }

  if (PRINTF_MAX_INTEGRAL_DIGITS_FOR_DECIMAL < 6) {
    printing_check("1e+6", "==", invokable::sprintf, buffer, base_buffer_size, "%.0f", (double) ((int64_t)1 * 1000 * 1000 ) );
  }
  else {
    printing_check("1000000", "==", invokable::sprintf, buffer, base_buffer_size, "%.0f", (double) ((int64_t)1 * 1000 * 1000 ) );
  }

  if (PRINTF_MAX_INTEGRAL_DIGITS_FOR_DECIMAL < 9) {
    printing_check("1e+9", "==", invokable::sprintf, buffer, base_buffer_size, "%.0f", (double) ((int64_t)1 * 1000 * 1000 * 1000 ) );
  }
  else {
    printing_check("1000000000", "==", invokable::sprintf, buffer, base_buffer_size, "%.0f", (double) ((int64_t)1 * 1000 * 1000 * 1000 ) );
  }

  if (PRINTF_MAX_INTEGRAL_DIGITS_FOR_DECIMAL < 12) {
    printing_check("1e+12", "==", invokable::sprintf, buffer, base_buffer_size, "%.0f", (double) ((int64_t)1 * 1000 * 1000 * 1000 * 1000) );
  }
  else {
    printing_check("1000000000000", "==", invokable::sprintf, buffer, base_buffer_size, "%.0f", (double) ((int64_t)1 * 1000 * 1000 * 1000 * 1000) );
  }

  if (PRINTF_MAX_INTEGRAL_DIGITS_FOR_DECIMAL < 15) {
    printing_check("1e+15", "==", invokable::sprintf, buffer, base_buffer_size, "%.0f", (double) ((int64_t)1 * 1000 * 1000 * 1000 * 1000 * 1000) );
  }
  else {
    printing_check("1000000000000000", "==", invokable::sprintf, buffer, base_buffer_size, "%.0f", (double) ((int64_t)1 * 1000 * 1000 * 1000 * 1000 * 1000) );
  }
  printing_check("34", "==", invokable::sprintf, buffer, base_buffer_size, "%.0f", 34.1415354);
  printing_check("1", "==", invokable::sprintf, buffer, base_buffer_size, "%.0f", 1.3);
  printing_check("2", "==", invokable::sprintf, buffer, base_buffer_size, "%.0f", 1.55);
  printing_check("1.6", "==", invokable::sprintf, buffer, base_buffer_size, "%.1f", 1.64);
  printing_check("42.90", "==", invokable::sprintf, buffer, base_buffer_size, "%.2f", 42.8952);
  printing_check("42.895200000", "==", invokable::sprintf, buffer, base_buffer_size, "%.9f", 42.8952);
  printing_check("42.8952230000", "==", invokable::sprintf, buffer, base_buffer_size, "%.10f", 42.895223);
  printing_check("42.895223123457", "==", invokable::sprintf, buffer, base_buffer_size, "%.12f", 42.89522312345678);
  printing_check("42477.371093750000000", "==", invokable::sprintf, buffer, base_buffer_size, "%020.15f", 42477.37109375);
  printing_check("42.895223876543", "==", invokable::sprintf, buffer, base_buffer_size, "%.12f", 42.89522387654321);
  printing_check(" 42.90", "==", invokable::sprintf, buffer, base_buffer_size, "%6.2f", 42.8952);
  printing_check("+42.90", "==", invokable::sprintf, buffer, base_buffer_size, "%+6.2f", 42.8952);
  printing_check("+42.9", "==", invokable::sprintf, buffer, base_buffer_size, "%+5.1f", 42.9252);
  printing_check("42.500000", "==", invokable::sprintf, buffer, base_buffer_size, "%f", 42.5);
  printing_check("42.5", "==", invokable::sprintf, buffer, base_buffer_size, "%.1f", 42.5);
  printing_check("42167.000000", "==", invokable::sprintf, buffer, base_buffer_size, "%f", 42167.0);
  printing_check("-12345.987654321", "==", invokable::sprintf, buffer, base_buffer_size, "%.9f", -12345.987654321);
  printing_check("4.0", "==", invokable::sprintf, buffer, base_buffer_size, "%.1f", 3.999);
  printing_check("4", "==", invokable::sprintf, buffer, base_buffer_size, "%.0f", 3.5);
  printing_check("4", "==", invokable::sprintf, buffer, base_buffer_size, "%.0f", 4.5);
  printing_check("3", "==", invokable::sprintf, buffer, base_buffer_size, "%.0f", 3.49);
  printing_check("3.5", "==", invokable::sprintf, buffer, base_buffer_size, "%.1f", 3.49);
  printing_check("a0.5  ", "==", invokable::sprintf, buffer, base_buffer_size, "a%-5.1f", 0.5);
  printing_check("a0.5  end", "==", invokable::sprintf, buffer, base_buffer_size, "a%-5.1fend", 0.5);

#ifndef PRINTF_DISABLE_SUPPORT_EXPONENTIAL
  printing_check("12345.7", "==", invokable::sprintf, buffer, base_buffer_size, "%G", 12345.678);
  printing_check("12345.68", "==", invokable::sprintf, buffer, base_buffer_size, "%.7G", 12345.678);
  printing_check("1.2346E+08", "==", invokable::sprintf, buffer, base_buffer_size, "%.5G", 123456789.);
  printing_check("12345", "==", invokable::sprintf, buffer, base_buffer_size, "%.6G", 12345.);
  printing_check("  +1.235e+08", "==", invokable::sprintf, buffer, base_buffer_size, "%+12.4g", 123456789.);
  printing_check("0.0012", "==", invokable::sprintf, buffer, base_buffer_size, "%.2G", 0.001234);
  printing_check(" +0.001234", "==", invokable::sprintf, buffer, base_buffer_size, "%+10.4G", 0.001234);
  printing_check("+001.234e-05", "==", invokable::sprintf, buffer, base_buffer_size, "%+012.4g", 0.00001234);
  printing_check("-1.23e-308", "==", invokable::sprintf, buffer, base_buffer_size, "%.3g", -1.2345e-308);
  printing_check("+1.230E+308", "==", invokable::sprintf, buffer, base_buffer_size, "%+.3E", 1.23e+308);
#endif

  // out of range for float: should switch to exp notation if supported, else empty
#ifndef PRINTF_DISABLE_SUPPORT_EXPONENTIAL
  printing_check("1.0e+20", "==", invokable::sprintf, buffer, base_buffer_size, "%.1f", 1E20);
#else
  printing_check("", "==", invokable::sprintf, buffer, base_buffer_size, "%.1f", 1E20);
#endif

  // brute force float
  bool fail = false;
  std::stringstream str;
  str.precision(5);
  for (float i = -100000; i < 100000; i += 1) {
   invoke_on_device(invokable::sprintf, buffer, "%.5f", (double)(i / 10000));
    str.str("");
    str << std::fixed << i / 10000;
    fail = fail || !!strcmp(buffer, str.str().c_str());
  }
  if (fail) {
    std::cerr << "sprintf(\"" << "%.5f\" (double)(i / 10000)) failed." << std::endl;
    exit(EXIT_FAILURE);
  }


#ifndef PRINTF_DISABLE_SUPPORT_EXPONENTIAL
  // brute force exp
  str.setf(std::ios::scientific, std::ios::floatfield);
  for (float i = -1e20; i < (float) 1e20; i += (float) 1e15) {
   invoke_on_device(invokable::sprintf, buffer, "%.5f", (double) i);
    str.str("");
    str << i;
    fail = fail || !!strcmp(buffer, str.str().c_str());
  }
  if (fail) {
    std::cerr << "sprintf(\"" << "%.5f\" (double) i) failed." << std::endl;
    exit(EXIT_FAILURE);
  }
#endif
}


void testcase_types() {
  char buffer[base_buffer_size];
  printing_check("0", "==", invokable::sprintf, buffer, base_buffer_size, "%i", 0);
  printing_check("1234", "==", invokable::sprintf, buffer, base_buffer_size, "%i", 1234);
  printing_check("32767", "==", invokable::sprintf, buffer, base_buffer_size, "%i", 32767);
  printing_check("-32767", "==", invokable::sprintf, buffer, base_buffer_size, "%i", -32767);
  printing_check("30", "==", invokable::sprintf, buffer, base_buffer_size, "%li", 30L);
  printing_check("-2147483647", "==", invokable::sprintf, buffer, base_buffer_size, "%li", -2147483647L);
  printing_check("2147483647", "==", invokable::sprintf, buffer, base_buffer_size, "%li", 2147483647L);
  printing_check("30", "==", invokable::sprintf, buffer, base_buffer_size, "%lli", 30LL);
  printing_check("-9223372036854775807", "==", invokable::sprintf, buffer, base_buffer_size, "%lli", -9223372036854775807LL);
  printing_check("9223372036854775807", "==", invokable::sprintf, buffer, base_buffer_size, "%lli", 9223372036854775807LL);
  printing_check("100000", "==", invokable::sprintf, buffer, base_buffer_size, "%lu", 100000L);
  printing_check("4294967295", "==", invokable::sprintf, buffer, base_buffer_size, "%lu", 0xFFFFFFFFL);
  printing_check("281474976710656", "==", invokable::sprintf, buffer, base_buffer_size, "%llu", 281474976710656LLU);
  printing_check("18446744073709551615", "==", invokable::sprintf, buffer, base_buffer_size, "%llu", 18446744073709551615LLU);
  printing_check("2147483647", "==", invokable::sprintf, buffer, base_buffer_size, "%zu", (size_t)2147483647UL);
  printing_check("2147483647", "==", invokable::sprintf, buffer, base_buffer_size, "%zd", (size_t)2147483647UL);
  printing_check("-2147483647", "==", invokable::sprintf, buffer, base_buffer_size, "%zi", (ssize_t)-2147483647L);
  printing_check("165140", "==", invokable::sprintf, buffer, base_buffer_size, "%o", 60000);
  printing_check("57060516", "==", invokable::sprintf, buffer, base_buffer_size, "%lo", 12345678L);
  printing_check("12345678", "==", invokable::sprintf, buffer, base_buffer_size, "%lx", 0x12345678L);
  printing_check("1234567891234567", "==", invokable::sprintf, buffer, base_buffer_size, "%llx", 0x1234567891234567LLU);
  printing_check("abcdefab", "==", invokable::sprintf, buffer, base_buffer_size, "%lx", 0xabcdefabL);
  printing_check("ABCDEFAB", "==", invokable::sprintf, buffer, base_buffer_size, "%lX", 0xabcdefabL);
  printing_check("v", "==", invokable::sprintf, buffer, base_buffer_size, "%c", 'v');
  printing_check("wv", "==", invokable::sprintf, buffer, base_buffer_size, "%cv", 'w');
  printing_check("A Test", "==", invokable::sprintf, buffer, base_buffer_size, "%s", mds("A Test"));
  printing_check("255", "==", invokable::sprintf, buffer, base_buffer_size, "%hhu", (unsigned char) 0xFFU);
  printing_check("4660", "==", invokable::sprintf, buffer, base_buffer_size, "%hu", (unsigned short) 0x1234u);
  printing_check("Test100 65535", "==", invokable::sprintf, buffer, base_buffer_size, "%s%hhi %hu", mds("Test"), (char) 100, (unsigned short) 0xFFFF);
  printing_check("a", "==", invokable::sprintf, buffer, base_buffer_size, "%tx", &buffer[10] - &buffer[0]);
  printing_check("-2147483647", "==", invokable::sprintf, buffer, base_buffer_size, "%ji", (intmax_t)-2147483647L);
}

#ifdef TEST_WITH_NON_STANDARD_FORMAT_STRINGS
void testcase_types__non_standard_format() {
  char buffer[base_buffer_size];
  printing_check("1110101001100000", "==", invokable::sprintf, buffer, base_buffer_size, "%b", 60000);
  printing_check("101111000110000101001110", "==", invokable::sprintf, buffer, base_buffer_size, "%lb", 12345678L);
}
#endif

void testcase_pointer() {
  char buffer[base_buffer_size];

  if (sizeof(void*) == 4U) {
    printing_check("0x00001234", "==", invokable::sprintf, buffer, base_buffer_size, "%p", (void*)0x1234U);
  }
  else {
    printing_check("0x0000000000001234", "==", invokable::sprintf, buffer, base_buffer_size, "%p", (void*)0x1234U);
  }

  if (sizeof(void*) == 4U) {
    printing_check("0x12345678", "==", invokable::sprintf, buffer, base_buffer_size, "%p", (void*)0x12345678U);
  }
  else {
    printing_check("0x0000000012345678", "==", invokable::sprintf, buffer, base_buffer_size, "%p", (void*)0x12345678U);
  }

  if (sizeof(void*) == 4U) {
    printing_check("0x12345678-0x7edcba98", "==", invokable::sprintf, buffer, base_buffer_size, "%p-%p", (void*)0x12345678U, (void*)0x7EDCBA98U);
  }
  else {
    printing_check("0x0000000012345678-0x000000007edcba98", "==", invokable::sprintf, buffer, base_buffer_size, "%p-%p", (void*)0x12345678U, (void*)0x7EDCBA98U);
  }

  if (sizeof(uintptr_t) == sizeof(uint64_t)) {
   printing_check("0x00000000ffffffff", "==", invokable::sprintf, buffer, base_buffer_size, "%p", (void*)(uintptr_t)0xFFFFFFFFU);
  }
  else {
   printing_check("0xffffffff", "==", invokable::sprintf, buffer, base_buffer_size, "%p", (void*)(uintptr_t)0xFFFFFFFFU);
  }
  printing_check("(nil)", "==", invokable::sprintf, buffer, base_buffer_size, "%p", NULL);
}

#ifdef TEST_WITH_NON_STANDARD_FORMAT_STRINGS
void testcase_unknown_flag__non_standard_format() {
  char buffer[base_buffer_size];
  printing_check("kmarco", "==", invokable::sprintf, buffer, base_buffer_size, "%kmarco", 42, 37);
}
#endif

void testcase_string_length() {
  char buffer[base_buffer_size];
  printing_check("This", "==", invokable::sprintf, buffer, base_buffer_size, "%.4s", mds("This is a test"));
  printing_check("test", "==", invokable::sprintf, buffer, base_buffer_size, "%.4s", mds("test"));
  printing_check("123", "==", invokable::sprintf, buffer, base_buffer_size, "%.7s", mds("123"));
  printing_check("", "==", invokable::sprintf, buffer, base_buffer_size, "%.7s", mds(""));
  printing_check("1234ab", "==", invokable::sprintf, buffer, base_buffer_size, "%.4s%.2s", mds("123456"), mds("abcdef"));
  printing_check("123", "==", invokable::sprintf, buffer, base_buffer_size, "%.*s", 3, mds("123456"));

DISABLE_WARNING_PUSH
DISABLE_WARNING_PRINTF_FORMAT_OVERFLOW
  printing_check("(null)", "==", invokable::sprintf, buffer, base_buffer_size, "%.*s", 3, (const char*) NULL);
DISABLE_WARNING_POP
}

#ifdef TEST_WITH_NON_STANDARD_FORMAT_STRINGS
void testcase_string_length__non_standard_format() {
  char buffer[base_buffer_size];
  printing_check(".2s", "==", invokable::sprintf, buffer, base_buffer_size, "%.4.2s", mds("123456"));
}
#endif


void testcase_buffer_length() {
  char buffer[base_buffer_size];

  printing_and_ret_check(4, nullptr, "==", invokable::snprintf, nullptr, 10, "%s", mds("Test"));

  buffer[0] = (char)0xA5;
  printing_and_ret_check(4, "", "==", invokable::snprintf, buffer, (size_t) 0, "%s", mds("Test"));
  if (buffer[0] != (char)0xA5) {
    std::cerr << "snprintf snprintf(buffer, 0, \"%s\", \"Test\") modified characters, when it should not have." << std::endl;
    exit(EXIT_FAILURE);
  }
  buffer[0] = (char)0xCC;
  printing_check("", "==", invokable::snprintf, buffer, 1, "%s", mds("Test"));
  printing_check("H", "==", invokable::snprintf, buffer, 2, "%s", mds("Hello"));

DISABLE_WARNING_PUSH
DISABLE_WARNING_PRINTF_FORMAT_OVERFLOW
  printing_check("(", "==", invokable::snprintf, buffer, 2, "%s", NULL);
DISABLE_WARNING_POP
}


void testcase_ret_value() {
  char buffer[base_buffer_size];
  printing_and_ret_check( 5, "01234",     "==", invokable::snprintf, buffer,  6, "0%s", mds("1234"));
  printing_and_ret_check( 6, "01234",     "==", invokable::snprintf, buffer,  6, "0%s", mds("12345"));
  printing_and_ret_check( 8, "01234",     "==", invokable::snprintf, buffer,  6, "0%s", mds("1234567"));
DISABLE_WARNING_PUSH
DISABLE_WARNING_PRINTF_FORMAT_OVERFLOW
  printing_and_ret_check( 7, "0(nul",     "==", invokable::snprintf, buffer,  6, "0%s", (const char*) NULL);
DISABLE_WARNING_POP
  printing_and_ret_check(12, "hello, wo", "==", invokable::snprintf, buffer, 10, "hello, world");
  printing_and_ret_check( 5, "10",        "==", invokable::snprintf, buffer,  3, "%d", 10000);
}

void testcase_misc() {
  char buffer[base_buffer_size];
  printing_check("53000atest-20 bit", "==", invokable::sprintf, buffer, base_buffer_size, "%u%u%ctest%d %s", 5, 3000, 'a', -20, mds("bit"));
  printing_check("0.33", "==", invokable::sprintf, buffer, base_buffer_size, "%.*f", 2, 0.33333333);
  printing_check("1", "==", invokable::sprintf, buffer, base_buffer_size, "%.*d", -1, 1);
  printing_check("foo", "==", invokable::sprintf, buffer, base_buffer_size, "%.3s", mds("foobar"));
  printing_check(" ", "==", invokable::sprintf, buffer, base_buffer_size, "% .0d", 0);
  printing_check("     00004", "==", invokable::sprintf, buffer, base_buffer_size, "%10.5d", 4);
  printing_check("hi x", "==", invokable::sprintf, buffer, base_buffer_size, "%*sx", -3, mds("hi"));

#ifndef PRINTF_DISABLE_SUPPORT_EXPONENTIAL
  printing_check("0.33", "==", invokable::sprintf, buffer, base_buffer_size, "%.*g", 2, 0.33333333);
  printing_check("3.33e-01", "==", invokable::sprintf, buffer, base_buffer_size, "%.*e", 2, 0.33333333);
#endif
}

#ifdef TEST_WITH_NON_STANDARD_FORMAT_STRINGS
DISABLE_WARNING_POP
#endif

int main()
{
  testcase_fctprintf();
  testcase_vfctprintf();
  testcase_snprintf();
  testcase_vsprintf();
  testcase_vsnprintf();
  testcase_simple_sprintf();
  testcase_space_flag();
#ifdef TEST_WITH_NON_STANDARD_FORMAT_STRINGS
  testcase_space_flag__non_standard_format();
#endif
  testcase_plus_flag();
  testcase_0_flag();
  testcase_minus_flag();
#ifdef TEST_WITH_NON_STANDARD_FORMAT_STRINGS
  testcase_minus_flag__non_standard_format();
#endif
  testcase_hash_flag();

  // FIXME: this is not set!
#ifdef TEST_WITH_NON_STANDARD_FORMAT_STRINGS
  testcase_hash_flag__non_standard_format();
#endif
  testcase_specifier();
  testcase_width();
  std::cout << "ok" << std::endl;
  testcase_width_20();
  testcase_width_star_20();
  testcase_width_minus_20();
#ifdef TEST_WITH_NON_STANDARD_FORMAT_STRINGS
  testcase_width_0_minus_20();
#endif
  testcase_padding_20();
  testcase_padding_dot_20();
#ifdef TEST_WITH_NON_STANDARD_FORMAT_STRINGS
  testcase_padding_hash_020__non_standard_format();
#endif
  testcase_padding_hash_020();
#ifdef TEST_WITH_NON_STANDARD_FORMAT_STRINGS
  testcase_padding_hash_20__non_standard_format();
#endif
  testcase_padding_hash_20();
  testcase_padding_20_dot_5();
  testcase_padding_neg_numbers();

  testcase_float_padding_neg_numbers();
  testcase_length();
#ifdef TEST_WITH_NON_STANDARD_FORMAT_STRINGS
  testcase_length__non_standard_format();
#endif
  testcase_float();
  testcase_types();
#ifdef TEST_WITH_NON_STANDARD_FORMAT_STRINGS
  testcase_types__non_standard_format();
#endif
  testcase_pointer();
#ifdef TEST_WITH_NON_STANDARD_FORMAT_STRINGS
  testcase_unknown_flag__non_standard_format();
#endif
  testcase_string_length();
#ifdef TEST_WITH_NON_STANDARD_FORMAT_STRINGS
  testcase_string_length__non_standard_format();
#endif
  testcase_buffer_length();
  testcase_ret_value();
  testcase_misc();
}
