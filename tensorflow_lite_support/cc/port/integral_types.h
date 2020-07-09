/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_SUPPORT_CC_PORT_INTEGRAL_TYPES_H_
#define TENSORFLOW_LITE_SUPPORT_CC_PORT_INTEGRAL_TYPES_H_

// TODO: <sys/types.h> is not portable C. Take a close look at this when we add
// mobile support.
#include <sys/types.h>
#include <cstdint>

typedef signed char        schar;
typedef int8_t             int8;
typedef int16_t            int16;
typedef int32_t            int32;
typedef int64_t            int64;

typedef uint8_t            uint8;
typedef uint16_t           uint16;
typedef uint32_t           uint32;
typedef uint32_t           char32;
typedef uint64_t           uint64;

typedef unsigned long      uword_t;

#define GG_LONGLONG(x) x##LL
#define GG_ULONGLONG(x) x##ULL
#define GG_LL_FORMAT "ll"  // As in "%lld". Note that "q" is poor form also.
#define GG_LL_FORMAT_W L"ll"

typedef uint64 Fprint;
static const Fprint kIllegalFprint = 0;
static const Fprint kMaxFprint = GG_ULONGLONG(0xFFFFFFFFFFFFFFFF);

#endif  // TENSORFLOW_LITE_SUPPORT_CC_PORT_INTEGRAL_TYPES_H_
