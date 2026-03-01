// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef EFANNA2E_DISTANCE_H
#define EFANNA2E_DISTANCE_H

#include <immintrin.h>
#include <x86intrin.h>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace efanna2e {

enum Metric { L2 = 0, INNER_PRODUCT = 1, FAST_L2 = 2, COSINE = 3 };

class DistanceBase {
public:
  inline static std::atomic<size_t> compare_count{0};
  static void reset_compare_count() {
    compare_count.store(0, std::memory_order_relaxed);
  }
};

template <typename T>
class Distance {
public:
  virtual float compare(const T* a, const T* b, uint32_t length) const = 0;
  virtual ~Distance() {}
};

// ---------------------- helpers ----------------------
#if defined(__AVX2__)
static inline int32_t hsum_epi32_avx(__m256i v) {
  __m128i vlow  = _mm256_castsi256_si128(v);
  __m128i vhigh = _mm256_extracti128_si256(v, 1);
  __m128i sum128 = _mm_add_epi32(vlow, vhigh);
  __m128i shuf = _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2,3,0,1));
  sum128 = _mm_add_epi32(sum128, shuf);
  shuf = _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1,0,3,2));
  sum128 = _mm_add_epi32(sum128, shuf);
  return _mm_cvtsi128_si32(sum128);
}
#endif

// =====================================================
// L2
// =====================================================
template <typename T>
class DistanceL2 : public Distance<T> {
public:
  float compare(const T* a, const T* b, uint32_t length) const override {
    // DistanceBase::compare_count.fetch_add(1, std::memory_order_relaxed);

    // ---- fast path: float ----
    if constexpr (std::is_same<T,float>::value) {
#if defined(__AVX2__) && defined(__FMA__)
      __m256 acc = _mm256_setzero_ps();
      uint32_t i = 0;
      for (; i + 16 <= length; i += 16) {
        __m256 a0=_mm256_loadu_ps(a+i),   b0=_mm256_loadu_ps(b+i);
        __m256 d0=_mm256_sub_ps(a0,b0);   acc=_mm256_fmadd_ps(d0,d0,acc);
        __m256 a1=_mm256_loadu_ps(a+i+8), b1=_mm256_loadu_ps(b+i+8);
        __m256 d1=_mm256_sub_ps(a1,b1);   acc=_mm256_fmadd_ps(d1,d1,acc);
      }
      if (i + 8 <= length) {
        __m256 a0=_mm256_loadu_ps(a+i), b0=_mm256_loadu_ps(b+i);
        __m256 d0=_mm256_sub_ps(a0,b0); acc=_mm256_fmadd_ps(d0,d0,acc);
        i += 8;
      }
      alignas(32) float tmp[8];
      _mm256_storeu_ps(tmp, acc);
      float res = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
      for (; i<length; ++i){ float d=a[i]-b[i]; res += d*d; }
      return res;
#elif defined(__SSE2__)
      __m128 acc = _mm_setzero_ps();
      uint32_t i = 0;
      for (; i + 16 <= length; i += 16) {
        __m128 a0=_mm_loadu_ps(a+i),   b0=_mm_loadu_ps(b+i);
        __m128 d0=_mm_sub_ps(a0,b0);   acc=_mm_add_ps(acc,_mm_mul_ps(d0,d0));
        __m128 a1=_mm_loadu_ps(a+i+4), b1=_mm_loadu_ps(b+i+4);
        __m128 d1=_mm_sub_ps(a1,b1);   acc=_mm_add_ps(acc,_mm_mul_ps(d1,d1));
        __m128 a2=_mm_loadu_ps(a+i+8), b2=_mm_loadu_ps(b+i+8);
        __m128 d2=_mm_sub_ps(a2,b2);   acc=_mm_add_ps(acc,_mm_mul_ps(d2,d2));
        __m128 a3=_mm_loadu_ps(a+i+12),b3=_mm_loadu_ps(b+i+12);
        __m128 d3=_mm_sub_ps(a3,b3);   acc=_mm_add_ps(acc,_mm_mul_ps(d3,d3));
      }
      if (i + 4 <= length) {
        __m128 a0=_mm_loadu_ps(a+i), b0=_mm_loadu_ps(b+i);
        __m128 d0=_mm_sub_ps(a0,b0); acc=_mm_add_ps(acc,_mm_mul_ps(d0,d0));
        i += 4;
      }
      alignas(16) float t[4]; _mm_storeu_ps(t, acc);
      float res = t[0]+t[1]+t[2]+t[3];
      for (; i<length; ++i){ float d=a[i]-b[i]; res += d*d; }
      return res;
#else
      float res = 0.f;
      for (uint32_t i=0;i<length;++i){ float d=a[i]-b[i]; res+=d*d; }
      return res;
#endif
    }

    // ---- fast path: 8-bit integral (int8/char/signed char/uint8) ----
    else if constexpr (std::is_integral<T>::value && sizeof(T)==1) {
#if defined(__AVX2__)
      const uint8_t* pa = reinterpret_cast<const uint8_t*>(a);
      const uint8_t* pb = reinterpret_cast<const uint8_t*>(b);
      __m256i acc32 = _mm256_setzero_si256();
      uint32_t i = 0;

      // process 32 bytes per iter (two 16B chunks)
      for (; i + 32 <= length; i += 32) {
        // lower 16 bytes
        __m128i a128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pa + i));
        __m128i b128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pb + i));
        __m256i a16, b16;
        if constexpr (std::is_signed<T>::value) {
          a16 = _mm256_cvtepi8_epi16(a128);
          b16 = _mm256_cvtepi8_epi16(b128);
        } else {
          a16 = _mm256_cvtepu8_epi16(a128);
          b16 = _mm256_cvtepu8_epi16(b128);
        }
        __m256i d16 = _mm256_sub_epi16(a16, b16);
        __m256i sq0 = _mm256_madd_epi16(d16, d16); // pairwise squares -> i32
        acc32 = _mm256_add_epi32(acc32, sq0);

        // upper 16 bytes
        __m128i a128h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pa + i + 16));
        __m128i b128h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pb + i + 16));
        if constexpr (std::is_signed<T>::value) {
          a16 = _mm256_cvtepi8_epi16(a128h);
          b16 = _mm256_cvtepi8_epi16(b128h);
        } else {
          a16 = _mm256_cvtepu8_epi16(a128h);
          b16 = _mm256_cvtepu8_epi16(b128h);
        }
        d16 = _mm256_sub_epi16(a16, b16);
        __m256i sq1 = _mm256_madd_epi16(d16, d16);
        acc32 = _mm256_add_epi32(acc32, sq1);
      }

      // tail 16 bytes
      if (i + 16 <= length) {
        __m128i a128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pa + i));
        __m128i b128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pb + i));
        __m256i a16, b16;
        if constexpr (std::is_signed<T>::value) {
          a16 = _mm256_cvtepi8_epi16(a128);
          b16 = _mm256_cvtepi8_epi16(b128);
        } else {
          a16 = _mm256_cvtepu8_epi16(a128);
          b16 = _mm256_cvtepu8_epi16(b128);
        }
        __m256i d16 = _mm256_sub_epi16(a16, b16);
        __m256i sq = _mm256_madd_epi16(d16, d16);
        acc32 = _mm256_add_epi32(acc32, sq);
        i += 16;
      }

      int32_t sum = hsum_epi32_avx(acc32);
      // scalar tail
      for (; i<length; ++i) {
        int da = std::is_signed<T>::value ? int(reinterpret_cast<const int8_t*>(a)[i])
                                          : int(reinterpret_cast<const uint8_t*>(a)[i]);
        int db = std::is_signed<T>::value ? int(reinterpret_cast<const int8_t*>(b)[i])
                                          : int(reinterpret_cast<const uint8_t*>(b)[i]);
        int d = da - db; sum += d * d;
      }
      return static_cast<float>(sum);
#elif defined(__SSE2__)
      // SSE2 fallback: 16 bytes per iter
      const uint8_t* pa = reinterpret_cast<const uint8_t*>(a);
      const uint8_t* pb = reinterpret_cast<const uint8_t*>(b);
      __m128i acc32 = _mm_setzero_si128();
      uint32_t i = 0;
      for (; i + 16 <= length; i += 16) {
        __m128i a128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pa + i));
        __m128i b128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pb + i));
        __m128i a16, b16;
        if constexpr (std::is_signed<T>::value) {
          a16 = _mm_cvtepi8_epi16(a128);
          b16 = _mm_cvtepi8_epi16(b128);
        } else {
          a16 = _mm_cvtepu8_epi16(a128);
          b16 = _mm_cvtepu8_epi16(b128);
        }
        __m128i d16 = _mm_sub_epi16(a16, b16);
        __m128i sq = _mm_madd_epi16(d16, d16); // 8 lanes -> i32
        acc32 = _mm_add_epi32(acc32, sq);
      }
      // horizontal sum
      alignas(16) int32_t t[4];
      _mm_storeu_si128(reinterpret_cast<__m128i*>(t), acc32);
      int32_t sum = t[0]+t[1]+t[2]+t[3];
      for (; i<length; ++i) {
        int da = std::is_signed<T>::value ? int(reinterpret_cast<const int8_t*>(a)[i])
                                          : int(reinterpret_cast<const uint8_t*>(a)[i]);
        int db = std::is_signed<T>::value ? int(reinterpret_cast<const int8_t*>(b)[i])
                                          : int(reinterpret_cast<const uint8_t*>(b)[i]);
        int d = da - db; sum += d * d;
      }
      return static_cast<float>(sum);
#else
      // portable scalar
      int32_t sum = 0;
      for (uint32_t i=0;i<length;++i) {
        int da = std::is_signed<T>::value ? int(reinterpret_cast<const int8_t*>(a)[i])
                                          : int(reinterpret_cast<const uint8_t*>(a)[i]);
        int db = std::is_signed<T>::value ? int(reinterpret_cast<const int8_t*>(b)[i])
                                          : int(reinterpret_cast<const uint8_t*>(b)[i]);
        int d = da - db; sum += d * d;
      }
      return static_cast<float>(sum);
#endif
    }

    // ---- fallback: cast to float and scalar ----
    else {
      float res = 0.f;
      for (uint32_t i=0;i<length;++i) {
        float d = static_cast<float>(a[i]) - static_cast<float>(b[i]);
        res += d*d;
      }
      return res;
    }
  }
};

// =====================================================
// Inner Product (return NEGATIVE dot-product as distance)
// =====================================================
template <typename T>
class DistanceInnerProduct : public Distance<T> {
public:
  float compare(const T* a, const T* b, uint32_t length) const override {
    // DistanceBase::compare_count.fetch_add(1, std::memory_order_relaxed);

    if constexpr (std::is_same<T,float>::value) {
#if defined(__AVX2__) && defined(__FMA__)
      __m256 acc = _mm256_setzero_ps();
      uint32_t i=0;
      for (; i+16<=length; i+=16) {
        __m256 a0=_mm256_loadu_ps(a+i),   b0=_mm256_loadu_ps(b+i);
        acc=_mm256_fmadd_ps(a0,b0,acc);
        __m256 a1=_mm256_loadu_ps(a+i+8), b1=_mm256_loadu_ps(b+i+8);
        acc=_mm256_fmadd_ps(a1,b1,acc);
      }
      if (i+8<=length) {
        __m256 a0=_mm256_loadu_ps(a+i), b0=_mm256_loadu_ps(b+i);
        acc=_mm256_fmadd_ps(a0,b0,acc); i+=8;
      }
      alignas(32) float t[8]; _mm256_storeu_ps(t, acc);
      float dot = t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
      for (; i<length; ++i) dot += a[i]*b[i];
      return -dot;
#elif defined(__SSE2__)
      __m128 acc = _mm_setzero_ps(); uint32_t i=0;
      for (; i+16<=length; i+=16) {
        __m128 a0=_mm_loadu_ps(a+i),   b0=_mm_loadu_ps(b+i);
        acc=_mm_add_ps(acc,_mm_mul_ps(a0,b0));
        __m128 a1=_mm_loadu_ps(a+i+4), b1=_mm_loadu_ps(b+i+4);
        acc=_mm_add_ps(acc,_mm_mul_ps(a1,b1));
        __m128 a2=_mm_loadu_ps(a+i+8), b2=_mm_loadu_ps(b+i+8);
        acc=_mm_add_ps(acc,_mm_mul_ps(a2,b2));
        __m128 a3=_mm_loadu_ps(a+i+12),b3=_mm_loadu_ps(b+i+12);
        acc=_mm_add_ps(acc,_mm_mul_ps(a3,b3));
      }
      if (i+4<=length) {
        __m128 a0=_mm_loadu_ps(a+i), b0=_mm_loadu_ps(b+i);
        acc=_mm_add_ps(acc,_mm_mul_ps(a0,b0)); i+=4;
      }
      alignas(16) float t[4]; _mm_storeu_ps(t, acc);
      float dot = t[0]+t[1]+t[2]+t[3];
      for (; i<length; ++i) dot += a[i]*b[i];
      return -dot;
#else
      float dot=0.f; for (uint32_t i=0;i<length;++i) dot+=a[i]*b[i];
      return -dot;
#endif
    }
    else if constexpr (std::is_integral<T>::value && sizeof(T)==1) {
#if defined(__AVX2__)
      const uint8_t* pa = reinterpret_cast<const uint8_t*>(a);
      const uint8_t* pb = reinterpret_cast<const uint8_t*>(b);
      __m256i acc32 = _mm256_setzero_si256();
      uint32_t i=0;

      for (; i+32<=length; i+=32) {
        __m128i a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pa+i));
        __m128i b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pb+i));
        __m256i a16, b16;
        if constexpr (std::is_signed<T>::value) {
          a16 = _mm256_cvtepi8_epi16(a0);
          b16 = _mm256_cvtepi8_epi16(b0);
        } else {
          a16 = _mm256_cvtepu8_epi16(a0);
          b16 = _mm256_cvtepu8_epi16(b0);
        }
        __m256i prod0 = _mm256_madd_epi16(a16, b16); // pairwise sum -> i32
        acc32 = _mm256_add_epi32(acc32, prod0);

        __m128i a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pa+i+16));
        __m128i b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pb+i+16));
        if constexpr (std::is_signed<T>::value) {
          a16 = _mm256_cvtepi8_epi16(a1);
          b16 = _mm256_cvtepi8_epi16(b1);
        } else {
          a16 = _mm256_cvtepu8_epi16(a1);
          b16 = _mm256_cvtepu8_epi16(b1);
        }
        __m256i prod1 = _mm256_madd_epi16(a16, b16);
        acc32 = _mm256_add_epi32(acc32, prod1);
      }
      if (i+16<=length) {
        __m128i a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pa+i));
        __m128i b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pb+i));
        __m256i a16, b16;
        if constexpr (std::is_signed<T>::value) {
          a16 = _mm256_cvtepi8_epi16(a0);
          b16 = _mm256_cvtepi8_epi16(b0);
        } else {
          a16 = _mm256_cvtepu8_epi16(a0);
          b16 = _mm256_cvtepu8_epi16(b0);
        }
        __m256i prod0 = _mm256_madd_epi16(a16, b16);
        acc32 = _mm256_add_epi32(acc32, prod0);
        i += 16;
      }
      int32_t sum = hsum_epi32_avx(acc32);
      for (; i<length; ++i) {
        int va = std::is_signed<T>::value ? int(reinterpret_cast<const int8_t*>(a)[i])
                                          : int(reinterpret_cast<const uint8_t*>(a)[i]);
        int vb = std::is_signed<T>::value ? int(reinterpret_cast<const int8_t*>(b)[i])
                                          : int(reinterpret_cast<const uint8_t*>(b)[i]);
        sum += va * vb;
      }
      return -static_cast<float>(sum);
#elif defined(__SSE2__)
      const uint8_t* pa = reinterpret_cast<const uint8_t*>(a);
      const uint8_t* pb = reinterpret_cast<const uint8_t*>(b);
      __m128i acc32 = _mm_setzero_si128();
      uint32_t i=0;
      for (; i+16<=length; i+=16) {
        __m128i a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pa+i));
        __m128i b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pb+i));
        __m128i a16, b16;
        if constexpr (std::is_signed<T>::value) {
          a16 = _mm_cvtepi8_epi16(a0);
          b16 = _mm_cvtepi8_epi16(b0);
        } else {
          a16 = _mm_cvtepu8_epi16(a0);
          b16 = _mm_cvtepu8_epi16(b0);
        }
        __m128i prod = _mm_madd_epi16(a16, b16);
        acc32 = _mm_add_epi32(acc32, prod);
      }
      alignas(16) int32_t t[4];
      _mm_storeu_si128(reinterpret_cast<__m128i*>(t), acc32);
      int32_t sum = t[0]+t[1]+t[2]+t[3];
      for (; i<length; ++i) {
        int va = std::is_signed<T>::value ? int(reinterpret_cast<const int8_t*>(a)[i])
                                          : int(reinterpret_cast<const uint8_t*>(a)[i]);
        int vb = std::is_signed<T>::value ? int(reinterpret_cast<const int8_t*>(b)[i])
                                          : int(reinterpret_cast<const uint8_t*>(b)[i]);
        sum += va * vb;
      }
      return -static_cast<float>(sum);
#else
      int32_t sum=0;
      for (uint32_t i=0;i<length;++i) {
        int va = std::is_signed<T>::value ? int(reinterpret_cast<const int8_t*>(a)[i])
                                          : int(reinterpret_cast<const uint8_t*>(a)[i]);
        int vb = std::is_signed<T>::value ? int(reinterpret_cast<const int8_t*>(b)[i])
                                          : int(reinterpret_cast<const uint8_t*>(b)[i]);
        sum += va*vb;
      }
      return -static_cast<float>(sum);
#endif
    }
    else {
      // generic fallback
      float dot = 0.f;
      for (uint32_t i=0;i<length;++i)
        dot += static_cast<float>(a[i]) * static_cast<float>(b[i]);
      return -dot;
    }
  }
};

// =====================================================
// Fast L2 (uses IP + norms), returns squared L2
// =====================================================
template <typename T>
class DistanceFastL2 : public DistanceInnerProduct<T> {
public:
  float compare(const T* a, const T* b, uint32_t length) const override {
    float na = norm(a, length);
    float nb = norm(b, length);
    float ip_neg = DistanceInnerProduct<T>::compare(a, b, length); // -dot
    return na + nb + 2.0f * ip_neg;
  }

  float norm(const T* a, uint32_t length) const {
    if constexpr (std::is_same<T,float>::value) {
#if defined(__AVX2__) && defined(__FMA__)
      __m256 acc=_mm256_setzero_ps(); uint32_t i=0;
      for (; i+16<=length; i+=16) {
        __m256 a0=_mm256_loadu_ps(a+i);   acc=_mm256_fmadd_ps(a0,a0,acc);
        __m256 a1=_mm256_loadu_ps(a+i+8); acc=_mm256_fmadd_ps(a1,a1,acc);
      }
      if (i+8<=length) { __m256 a0=_mm256_loadu_ps(a+i); acc=_mm256_fmadd_ps(a0,a0,acc); i+=8; }
      alignas(32) float t[8]; _mm256_storeu_ps(t, acc);
      float s=t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
      for (; i<length; ++i) s += a[i]*a[i];
      return s;
#elif defined(__SSE2__)
      __m128 acc=_mm_setzero_ps(); uint32_t i=0;
      for (; i+16<=length; i+=16) {
        __m128 a0=_mm_loadu_ps(a+i);   acc=_mm_add_ps(acc,_mm_mul_ps(a0,a0));
        __m128 a1=_mm_loadu_ps(a+i+4); acc=_mm_add_ps(acc,_mm_mul_ps(a1,a1));
        __m128 a2=_mm_loadu_ps(a+i+8); acc=_mm_add_ps(acc,_mm_mul_ps(a2,a2));
        __m128 a3=_mm_loadu_ps(a+i+12);acc=_mm_add_ps(acc,_mm_mul_ps(a3,a3));
      }
      if (i+4<=length){ __m128 a0=_mm_loadu_ps(a+i); acc=_mm_add_ps(acc,_mm_mul_ps(a0,a0)); i+=4; }
      alignas(16) float t[4]; _mm_storeu_ps(t, acc);
      float s=t[0]+t[1]+t[2]+t[3];
      for (; i<length; ++i) s += a[i]*a[i];
      return s;
#else
      float s=0.f; for (uint32_t i=0;i<length;++i) s+=a[i]*a[i]; return s;
#endif
    } else if constexpr (std::is_integral<T>::value && sizeof(T)==1) {
#if defined(__AVX2__)
      const uint8_t* pa = reinterpret_cast<const uint8_t*>(a);
      __m256i acc32 = _mm256_setzero_si256();
      uint32_t i=0;
      for (; i+32<=length; i+=32) {
        __m128i a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pa+i));
        __m256i a16 = std::is_signed<T>::value ? _mm256_cvtepi8_epi16(a0)
                                               : _mm256_cvtepu8_epi16(a0);
        __m256i sq0 = _mm256_madd_epi16(a16, a16);
        acc32 = _mm256_add_epi32(acc32, sq0);

        __m128i a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pa+i+16));
        a16 = std::is_signed<T>::value ? _mm256_cvtepi8_epi16(a1)
                                       : _mm256_cvtepu8_epi16(a1);
        __m256i sq1 = _mm256_madd_epi16(a16, a16);
        acc32 = _mm256_add_epi32(acc32, sq1);
      }
      if (i+16<=length) {
        __m128i a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pa+i));
        __m256i a16 = std::is_signed<T>::value ? _mm256_cvtepi8_epi16(a0)
                                               : _mm256_cvtepu8_epi16(a0);
        __m256i sq0 = _mm256_madd_epi16(a16, a16);
        acc32 = _mm256_add_epi32(acc32, sq0);
        i += 16;
      }
      int32_t sum = hsum_epi32_avx(acc32);
      for (; i<length; ++i) {
        int va = std::is_signed<T>::value ? int(reinterpret_cast<const int8_t*>(a)[i])
                                          : int(reinterpret_cast<const uint8_t*>(a)[i]);
        sum += va*va;
      }
      return static_cast<float>(sum);
#elif defined(__SSE2__)
      const uint8_t* pa = reinterpret_cast<const uint8_t*>(a);
      __m128i acc32 = _mm_setzero_si128(); uint32_t i=0;
      for (; i+16<=length; i+=16) {
        __m128i a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pa+i));
        __m128i a16 = std::is_signed<T>::value ? _mm_cvtepi8_epi16(a0)
                                               : _mm_cvtepu8_epi16(a0);
        __m128i sq0 = _mm_madd_epi16(a16, a16);
        acc32 = _mm_add_epi32(acc32, sq0);
      }
      alignas(16) int32_t t[4];
      _mm_storeu_si128(reinterpret_cast<__m128i*>(t), acc32);
      int32_t sum = t[0]+t[1]+t[2]+t[3];
      for (; i<length; ++i) {
        int va = std::is_signed<T>::value ? int(reinterpret_cast<const int8_t*>(a)[i])
                                          : int(reinterpret_cast<const uint8_t*>(a)[i]);
        sum += va*va;
      }
      return static_cast<float>(sum);
#else
      int32_t s=0; for (uint32_t i=0;i<length;++i) {
        int va = std::is_signed<T>::value ? int(reinterpret_cast<const int8_t*>(a)[i])
                                          : int(reinterpret_cast<const uint8_t*>(a)[i]);
        s += va*va;
      }
      return static_cast<float>(s);
#endif
    } else {
      float s=0.f; for (uint32_t i=0;i<length;++i){ float v=static_cast<float>(a[i]); s+=v*v; }
      return s;
    }
  }
};

// =====================================================
// Cosine (1 - cos) using int8/float SIMD
// =====================================================
template <typename T>
class DistanceCosine : public Distance<T> {
public:
  float compare(const T* a, const T* b, uint32_t length) const override {
    if constexpr (std::is_same<T,float>::value) {
      // reuse float dot + norms (AVX paths as above) — simplified scalar tail
#if defined(__AVX2__) && defined(__FMA__)
      __m256 acc_dot=_mm256_setzero_ps(), acc_a=_mm256_setzero_ps(), acc_b=_mm256_setzero_ps();
      uint32_t i=0;
      for (; i+16<=length; i+=16) {
        __m256 a0=_mm256_loadu_ps(a+i),   b0=_mm256_loadu_ps(b+i);
        acc_dot=_mm256_fmadd_ps(a0,b0,acc_dot);
        acc_a  =_mm256_fmadd_ps(a0,a0,acc_a);
        acc_b  =_mm256_fmadd_ps(b0,b0,acc_b);
        __m256 a1=_mm256_loadu_ps(a+i+8), b1=_mm256_loadu_ps(b+i+8);
        acc_dot=_mm256_fmadd_ps(a1,b1,acc_dot);
        acc_a  =_mm256_fmadd_ps(a1,a1,acc_a);
        acc_b  =_mm256_fmadd_ps(b1,b1,acc_b);
      }
      if (i+8<=length) {
        __m256 a0=_mm256_loadu_ps(a+i), b0=_mm256_loadu_ps(b+i);
        acc_dot=_mm256_fmadd_ps(a0,b0,acc_dot);
        acc_a  =_mm256_fmadd_ps(a0,a0,acc_a);
        acc_b  =_mm256_fmadd_ps(b0,b0,acc_b);
        i+=8;
      }
      alignas(32) float td[8], ta[8], tb[8];
      _mm256_storeu_ps(td, acc_dot);
      _mm256_storeu_ps(ta, acc_a);
      _mm256_storeu_ps(tb, acc_b);
      float dot=0, na=0, nb=0;
      for (int j=0;j<8;++j){ dot+=td[j]; na+=ta[j]; nb+=tb[j]; }
      for (; i<length; ++i) {
        dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i];
      }
      float denom = std::sqrt(na) * std::sqrt(nb);
      return 1.0f - (denom>0.f ? dot/denom : 0.f);
#else
      float dot=0, na=0, nb=0;
      for (uint32_t i=0;i<length;++i) {
        float va=a[i], vb=b[i];
        dot += va*vb; na += va*va; nb += vb*vb;
      }
      float denom = std::sqrt(na) * std::sqrt(nb);
      return 1.f - (denom>0.f ? dot/denom : 0.f);
#endif
    } else if constexpr (std::is_integral<T>::value && sizeof(T)==1) {
      // int8/uint8: compute dot and norms in int32 SIMD, then convert
#if defined(__AVX2__)
      const uint8_t* pa = reinterpret_cast<const uint8_t*>(a);
      const uint8_t* pb = reinterpret_cast<const uint8_t*>(b);
      __m256i acc_dot=_mm256_setzero_si256();
      __m256i acc_a  =_mm256_setzero_si256();
      __m256i acc_b  =_mm256_setzero_si256();
      uint32_t i=0;
      for (; i+32<=length; i+=32) {
        __m128i a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pa+i));
        __m128i b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pb+i));
        __m256i a16 = std::is_signed<T>::value ? _mm256_cvtepi8_epi16(a0)
                                               : _mm256_cvtepu8_epi16(a0);
        __m256i b16 = std::is_signed<T>::value ? _mm256_cvtepi8_epi16(b0)
                                               : _mm256_cvtepu8_epi16(b0);
        acc_dot = _mm256_add_epi32(acc_dot, _mm256_madd_epi16(a16,b16));
        acc_a   = _mm256_add_epi32(acc_a,   _mm256_madd_epi16(a16,a16));
        acc_b   = _mm256_add_epi32(acc_b,   _mm256_madd_epi16(b16,b16));

        __m128i a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pa+i+16));
        __m128i b1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pb+i+16));
        a16 = std::is_signed<T>::value ? _mm256_cvtepi8_epi16(a1) : _mm256_cvtepu8_epi16(a1);
        b16 = std::is_signed<T>::value ? _mm256_cvtepi8_epi16(b1) : _mm256_cvtepu8_epi16(b1);
        acc_dot = _mm256_add_epi32(acc_dot, _mm256_madd_epi16(a16,b16));
        acc_a   = _mm256_add_epi32(acc_a,   _mm256_madd_epi16(a16,a16));
        acc_b   = _mm256_add_epi32(acc_b,   _mm256_madd_epi16(b16,b16));
      }
      if (i+16<=length) {
        __m128i a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pa+i));
        __m128i b0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pb+i));
        __m256i a16 = std::is_signed<T>::value ? _mm256_cvtepi8_epi16(a0)
                                               : _mm256_cvtepu8_epi16(a0);
        __m256i b16 = std::is_signed<T>::value ? _mm256_cvtepi8_epi16(b0)
                                               : _mm256_cvtepu8_epi16(b0);
        acc_dot = _mm256_add_epi32(acc_dot, _mm256_madd_epi16(a16,b16));
        acc_a   = _mm256_add_epi32(acc_a,   _mm256_madd_epi16(a16,a16));
        acc_b   = _mm256_add_epi32(acc_b,   _mm256_madd_epi16(b16,b16));
        i += 16;
      }
      int32_t dot = hsum_epi32_avx(acc_dot);
      int32_t na  = hsum_epi32_avx(acc_a);
      int32_t nb  = hsum_epi32_avx(acc_b);
      for (; i<length; ++i) {
        int va = std::is_signed<T>::value ? int(reinterpret_cast<const int8_t*>(a)[i])
                                          : int(reinterpret_cast<const uint8_t*>(a)[i]);
        int vb = std::is_signed<T>::value ? int(reinterpret_cast<const int8_t*>(b)[i])
                                          : int(reinterpret_cast<const uint8_t*>(b)[i]);
        dot += va*vb; na += va*va; nb += vb*vb;
      }
      float fna = static_cast<float>(na);
      float fnb = static_cast<float>(nb);
      float fden = std::sqrt(fna) * std::sqrt(fnb);
      return 1.0f - (fden>0.f ? (static_cast<float>(dot) / fden) : 0.f);
#else
      int32_t dot=0, na=0, nb=0;
      for (uint32_t i=0;i<length;++i) {
        int va = std::is_signed<T>::value ? int(reinterpret_cast<const int8_t*>(a)[i])
                                          : int(reinterpret_cast<const uint8_t*>(a)[i]);
        int vb = std::is_signed<T>::value ? int(reinterpret_cast<const int8_t*>(b)[i])
                                          : int(reinterpret_cast<const uint8_t*>(b)[i]);
        dot += va*vb; na += va*va; nb += vb*vb;
      }
      float fna = static_cast<float>(na);
      float fnb = static_cast<float>(nb);
      float fden = std::sqrt(fna) * std::sqrt(fnb);
      return 1.0f - (fden>0.f ? (static_cast<float>(dot) / fden) : 0.f);
#endif
    } else {
      float dot=0, na=0, nb=0;
      for (uint32_t i=0;i<length;++i) { float va=static_cast<float>(a[i]); float vb=static_cast<float>(b[i]);
        dot+=va*vb; na+=va*va; nb+=vb*vb; }
      float denom = std::sqrt(na)*std::sqrt(nb);
      return 1.f - (denom>0.f ? dot/denom : 0.f);
    }
  }
};

} // namespace efanna2e
#endif // EFANNA2E_DISTANCE_H
