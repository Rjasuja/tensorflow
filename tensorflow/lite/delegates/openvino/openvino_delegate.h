/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_DELEGATES_OPENVINO_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_OPENVINO_DELEGATE_H_


#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/tools/logging.h"
#include "openvino_delegate_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct  {
    /* debug_level for the OpenVINO delegate*/
    int debug_level;

    /* path for the OpenVINO plugins */
    char* plugins_path;

    /* Device for OpenVINO to select 
        Currently we support CPU and NPU
    char* device_type*/;
} TfLiteOpenVINODelegateOptions;

TfLiteOpenVINODelegateOptions TfLiteOpenVINODelegateOptionsDefault();

TFL_CAPI_EXPORT TfLiteDelegate* TfLiteCreateOpenVINODelegate(TfLiteOpenVINODelegateOptions& options);

void TFL_CAPI_EXPORT TfLiteDeleteOpenVINODelegate(TfLiteDelegate* delegate);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_DELEGATES_OPENVINO_DELEGATE_H_