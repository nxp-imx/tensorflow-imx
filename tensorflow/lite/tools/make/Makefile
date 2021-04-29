# Make uses /bin/sh by default, which is incompatible with the bashisms seen
# below.
SHELL := /bin/bash

# Find where we're running from, so we can store generated files here.
ifeq ($(origin MAKEFILE_DIR), undefined)
	MAKEFILE_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
endif

# Try to figure out the host system
HOST_OS :=
ifeq ($(OS),Windows_NT)
	HOST_OS = windows
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		HOST_OS := linux
	endif
	ifeq ($(UNAME_S),Darwin)
		HOST_OS := osx
	endif
endif
HOST_OS := linux

HOST_ARCH := $(shell if uname -m | grep -q i[345678]86; then echo x86_32; else uname -m; fi)

# Override these on the make command line to target a specific architecture. For example:
# make -f tensorflow/lite/tools/make/Makefile TARGET=rpi TARGET_ARCH=armv7l
TARGET := $(HOST_OS)
TARGET_ARCH := $(HOST_ARCH)

INCLUDES := \
-I. \
-I$(MAKEFILE_DIR)/../../../../../ \
-I$(MAKEFILE_DIR)/../../../../../../ \
-I$(MAKEFILE_DIR)/downloads/ \
-I$(MAKEFILE_DIR)/downloads/eigen \
-I$(MAKEFILE_DIR)/downloads/absl \
-I$(MAKEFILE_DIR)/downloads/gemmlowp \
-I$(MAKEFILE_DIR)/downloads/ruy \
-I$(MAKEFILE_DIR)/downloads/neon_2_sse \
-I$(MAKEFILE_DIR)/downloads/farmhash/src \
-I$(MAKEFILE_DIR)/downloads/flatbuffers/include \
-I$(MAKEFILE_DIR)/downloads/fp16/include \
-I$(MAKEFILE_DIR)/downloads/cpuinfo \
-I$(MAKEFILE_DIR)/downloads/cpuinfo/include \
-I$(MAKEFILE_DIR)/downloads/cpuinfo/src \
-I$(MAKEFILE_DIR)/downloads/cpuinfo/deps/clog/include \
-I$(MAKEFILE_DIR)/downloads/googletest/googletest/include \
-I$(MAKEFILE_DIR)/downloads/googletest/googlemock/include \
-I$(MAKEFILE_DIR)/downloads/re2/ \
-I$(MAKEFILE_DIR)/../evaluation/stages \
-I$(MAKEFILE_DIR)/../../..tensorflow/core/lib/jpeg/ \
-I$(OBJDIR)

# This is at the end so any globally-installed frameworks like protobuf don't
# override local versions in the source tree.
INCLUDES += -I/usr/include

# These are the default libraries needed, but they can be added to or
# overridden by the platform-specific settings in target makefiles.
LIBS := \
-lstdc++ \
-lpthread \
-lm \
-lz \
-ldl \
-lprotobuf \
-ljpeg

# There are no rules for compiling objects for the host system (since we don't
# generate things like the protobuf compiler that require that), so all of
# these settings are for the target compiler.
CXXFLAGS += -O3 -DNDEBUG -DCPU_SETSIZE=__CPU_SETSIZE -DNNAPI_VERBOSE_VALIDATION -fPIC -Wno-narrowing
CXXFLAGS += $(EXTRA_CXXFLAGS)
CFLAGS := ${CXXFLAGS}
CXXFLAGS += --std=c++14
#LDOPTS := -L/usr/local/lib
ARFLAGS := -r
TARGET_TOOLCHAIN_PREFIX :=
CC_PREFIX :=

ifeq ($(HOST_OS),windows)
CXXFLAGS += -fext-numeric-literals -D__LITTLE_ENDIAN__
endif

ifeq ($(TARGET_ARCH),x86_64)
  ifeq ($(TARGET),linux)
    CXXFLAGS += -DTFLITE_HAVE_CPUINFO
    TFLITE_HAVE_CPUINFO := true
  endif
endif

# Auto-detect optimization opportunity if building natively.
ifeq ($(HOST_OS),$(TARGET))
ifeq ($(HOST_ARCH),$(TARGET_ARCH))
ifeq ($(TARGET_ARCH),armv7l)
ifneq ($(shell cat /proc/cpuinfo | grep Features | grep neon),)
  ifneq ($(shell cat /proc/cpuinfo | grep Features | grep vfpv4),)
    CXXFLAGS += -mfpu=neon-vfpv4
  else
    CXXFLAGS += -mfpu=neon
  endif
endif # ifeq ($(TARGET_ARCH),armv7l)
endif # ifeq ($(HOST_ARCH),$(TARGET_ARCH))
endif # ifeq ($(HOST_OS),$(TARGET))
endif

# This library is the main target for this makefile. It will contain a minimal
# runtime that can be linked in to other programs.
LIB_NAME := libtensorflow-lite.a

# Benchmark static library and binary
BENCHMARK_LIB_NAME := benchmark-lib.a
BENCHMARK_BINARY_NAME := benchmark_model
BENCHMARK_PERF_OPTIONS_BINARY_NAME := benchmark_model_performance_options

# A small example program that shows how to link against the library.
MINIMAL_SRCS := \
	tensorflow/lite/examples/minimal/minimal.cc

# What sources we want to compile, must be kept in sync with the main Bazel
# build files.

PROFILER_SRCS := \
	tensorflow/lite/profiling/memory_info.cc \
	tensorflow/lite/profiling/platform_profiler.cc \
	tensorflow/lite/profiling/time.cc

PROFILE_SUMMARIZER_SRCS := \
	tensorflow/lite/profiling/profile_summarizer.cc \
	tensorflow/lite/profiling/profile_summary_formatter.cc \
	tensorflow/core/util/stats_calculator.cc

CMD_LINE_TOOLS_SRCS := \
	tensorflow/lite/tools/command_line_flags.cc \
	tensorflow/lite/tools/tool_params.cc
	
LABEL_IMAGE_SRCS := \
	$(CMD_LINE_TOOLS_SRCS) \
	tensorflow/lite/examples/label_image/bitmap_helpers.cc \
	tensorflow/lite/examples/label_image/label_image.cc \
	tensorflow/lite/tools/evaluation/utils.cc

CORE_CC_ALL_SRCS := \
$(wildcard tensorflow/lite/*.cc) \
$(wildcard tensorflow/lite/*.c) \
$(wildcard tensorflow/lite/c/*.c) \
$(wildcard tensorflow/lite/c/*.cc) \
$(wildcard tensorflow/lite/core/*.cc) \
$(wildcard tensorflow/lite/core/api/*.cc) \
$(wildcard tensorflow/lite/experimental/resource/*.cc) \
$(wildcard tensorflow/lite/schema/schema_utils.cc) \
$(wildcard tensorflow/lite/tools/make/downloads/ruy/ruy/*.cc)
ifeq ($(TFLITE_HAVE_CPUINFO),true)
CORE_CC_ALL_SRCS += \
$(wildcard tensorflow/lite/tools/make/downloads/cpuinfo/src/*.c) \
$(wildcard tensorflow/lite/tools/make/downloads/cpuinfo/src/x86/*.c) \
$(wildcard tensorflow/lite/tools/make/downloads/cpuinfo/src/x86/linux/*.c) \
$(wildcard tensorflow/lite/tools/make/downloads/cpuinfo/src/x86/cache/*.c) \
$(wildcard tensorflow/lite/tools/make/downloads/cpuinfo/src/linux/*.c) \
$(wildcard tensorflow/lite/tools/make/downloads/cpuinfo/deps/clog/src/*.c)
endif
ifneq ($(BUILD_TYPE),micro)
CORE_CC_ALL_SRCS += \
$(wildcard tensorflow/lite/kernels/*.cc) \
$(wildcard tensorflow/lite/kernels/internal/*.cc) \
$(wildcard tensorflow/lite/kernels/internal/optimized/*.cc) \
$(wildcard tensorflow/lite/kernels/internal/reference/*.cc) \
$(wildcard tensorflow/lite/tools/optimize/sparsity/*.cc) \
$(PROFILER_SRCS) \
tensorflow/lite/tools/make/downloads/farmhash/src/farmhash.cc \
tensorflow/lite/tools/make/downloads/fft2d/fftsg.c \
tensorflow/lite/tools/make/downloads/fft2d/fftsg2d.c \
tensorflow/lite/tools/make/downloads/flatbuffers/src/util.cpp
CORE_CC_ALL_SRCS += \
	$(shell find tensorflow/lite/tools/make/downloads/absl/absl/ \
	             -type f -name \*.cc | grep -v test | grep -v benchmark | grep -v synchronization | grep -v debugging | grep -v hash | grep -v flags | grep -v random)
endif
# Remove any duplicates.
CORE_CC_ALL_SRCS := $(sort $(CORE_CC_ALL_SRCS))
CORE_CC_EXCLUDE_SRCS := \
$(wildcard tensorflow/lite/*test.cc) \
$(wildcard tensorflow/lite/*/*test.c) \
$(wildcard tensorflow/lite/*/*test.cc) \
$(wildcard tensorflow/lite/*/*/benchmark.cc) \
$(wildcard tensorflow/lite/*/*/example*.cc) \
$(wildcard tensorflow/lite/*/*/test*.cc) \
$(wildcard tensorflow/lite/*/*/*test.cc) \
$(wildcard tensorflow/lite/*/*/*tool.cc) \
$(wildcard tensorflow/lite/*/*/*/benchmark.cc) \
$(wildcard tensorflow/lite/*/*/*/example*.cc) \
$(wildcard tensorflow/lite/*/*/*/test*.cc) \
$(wildcard tensorflow/lite/*/*/*/*test.cc) \
$(wildcard tensorflow/lite/*/*/*/*tool.cc) \
$(wildcard tensorflow/lite/*/*/*/*/*/benchmark.cc) \
$(wildcard tensorflow/lite/*/*/*/*/*/example*.cc) \
$(wildcard tensorflow/lite/*/*/*/*/*/test*.cc) \
$(wildcard tensorflow/lite/*/*/*/*/*/*test.cc) \
$(wildcard tensorflow/lite/*/*/*/*/*/*tool.cc) \
$(wildcard tensorflow/lite/kernels/*test_main.cc) \
$(wildcard tensorflow/lite/kernels/*test_util*.cc) \
$(wildcard tensorflow/lite/tools/make/downloads/cpuinfo/src/*/mock*.c) \
tensorflow/lite/tflite_with_xnnpack.cc \
$(MINIMAL_SRCS)

UNIT_TEST_ALL_SRCS := \
$(wildcard tensorflow/lite/*test.cc) \
$(wildcard tensorflow/lite/kernels/*test.cc) \
$(wildcard tensorflow/lite/delegates/nnapi/*test.cc) \
$(wildcard tensorflow/lite/delegates/utils/*test.cc) \
$(wildcard tensorflow/lite/delegates/*test.cc) \
$(wildcard tensorflow/lite/c/*test.cc)

UNIT_TEST_EXCLUDE_SRCS := \
tensorflow/lite/model_xnnpack_test.cc \
tensorflow/lite/delegates/nnapi/nnapi_delegate_errno_test.cc \
tensorflow/lite/delegates/nnapi/nnapi_delegate_signed_quantization_test.cc \
tensorflow/lite/delegates/nnapi/nnapi_delegate_device_selection_test.cc

UNIT_TEST_SRCS := $(filter-out $(UNIT_TEST_EXCLUDE_SRCS), $(UNIT_TEST_ALL_SRCS))
UNIT_TEST_UTIL_SRCS := \
$(CMD_LINE_TOOLS_SRCS) \
$(wildcard tensorflow/lite/kernels/*test_util.cc) \
tensorflow/lite/tools/versioning/op_version.cc \
tensorflow/lite/kernels/internal/test_util.cc \
tensorflow/lite/kernels/acceleration_test_util_internal.cc \
$(wildcard tensorflow/lite/delegates/utils/dummy_delegate/*.cc ) \
tensorflow/lite/delegates/utils/simple_delegate.cc \
tensorflow/lite/delegates/utils.cc \
tensorflow/lite/delegates/interpreter_utils.cc \
tensorflow/lite/delegates/nnapi/acceleration_test_list.cc \
tensorflow/lite/delegates/nnapi/acceleration_test_util.cc \
tensorflow/lite/tools/evaluation/utils.cc \
tensorflow/lite/tools/optimize/model_utils.cc \
tensorflow/lite/tools/optimize/quantization_utils.cc \
tensorflow/lite/tools/optimize/operator_property.cc \
tensorflow/core/platform/default/logging.cc \
tensorflow/core/platform/default/env_time.cc

BUILD_WITH_MMAP ?= true
ifeq ($(BUILD_TYPE),micro)
	BUILD_WITH_MMAP=false
endif
ifeq ($(BUILD_TYPE),windows)
	BUILD_WITH_MMAP=false
endif
ifeq ($(BUILD_WITH_MMAP),true)
	CORE_CC_EXCLUDE_SRCS += tensorflow/lite/mmap_allocation_disabled.cc
else
	CORE_CC_EXCLUDE_SRCS += tensorflow/lite/mmap_allocation.cc
endif

BUILD_WITH_RUY ?= false
ifeq ($(TARGET_ARCH),aarch64)
	BUILD_WITH_RUY=true
endif
ifeq ($(BUILD_WITH_RUY),true)
  CXXFLAGS += -DTFLITE_WITH_RUY
endif

BUILD_WITH_RUY_PROFILER ?= false
ifeq ($(BUILD_WITH_RUY_PROFILER),true)
  CORE_CC_ALL_SRCS += tensorflow/lite/tools/make/downloads/ruy/ruy/profiler/instrumentation.cc
  CORE_CC_ALL_SRCS += tensorflow/lite/tools/make/downloads/ruy/ruy/profiler/profiler.cc
  CORE_CC_ALL_SRCS += tensorflow/lite/tools/make/downloads/ruy/ruy/profiler/treeview.cc
  CXXFLAGS += -DRUY_PROFILER
endif

# Not to include XNNPACK.
CXXFLAGS += -DTFLITE_WITHOUT_XNNPACK

BUILD_WITH_NNAPI ?= true
ifeq ($(BUILD_WITH_NNAPI),true)
	CORE_CC_ALL_SRCS += tensorflow/lite/delegates/nnapi/nnapi_delegate.cc
	CORE_CC_ALL_SRCS += tensorflow/lite/delegates/nnapi/quant_lstm_sup.cc
	CORE_CC_ALL_SRCS += tensorflow/lite/nnapi/nnapi_implementation.cc
	CORE_CC_ALL_SRCS += tensorflow/lite/nnapi/nnapi_util.cc
	UNIT_TEST_UTIL_SRCS += \
		tensorflow/lite/tools/delegates/nnapi_delegate_provider.cc  \
		tensorflow/lite/tools/delegates/default_execution_provider.cc
	LIBS += -lrt
else
	CORE_CC_ALL_SRCS += tensorflow/lite/delegates/nnapi/nnapi_delegate_disabled.cc
	CORE_CC_ALL_SRCS += tensorflow/lite/nnapi/nnapi_implementation_disabled.cc
endif

# Support VX_DELEGATE
BUILD_WITH_VX_DELEGATE ?= true
ifeq ($(BUILD_WITH_VX_DELEGATE), true)
    CXXFLAGS += -DTFLITE_WITH_VX
    INCLUDES += -I./tensorflow/lite/delegates/vx-delegate
    INCLUDES += -I./tensorflow/core/platform
    INCLUDES += -I./tensorflow/core/platform/default
    CORE_CC_ALL_SRCS += tensorflow/lite/delegates/vx-delegate/delegate_main.cc
    CORE_CC_ALL_SRCS += tensorflow/lite/delegates/vx-delegate/op_map.cc
    CORE_CC_ALL_SRCS += tensorflow/lite//delegates/vx-delegate/utils.cc
    CORE_CC_ALL_SRCS += tensorflow/core/platform/default/logging.cc
    CORE_CC_ALL_SRCS += tensorflow/core/platform/default/env_time.cc
    ifeq ($(TIM_VX_DIR), )
        $(warning  "TIM_VX_DIR not set, building from standard directories")
        # NO INCLUDE - try to locate include files in standard include paths
        LIBS += -ltim-vx
	else
        INCLUDES += -I$(TIM_VX_DIR)/include
        LIBS += -L$(TIM_VX_DIR)/build/src/tim/vx -l tim-vx
        LIBS += -L$(TIM_VX_DIR)/prebuilt-sdk/x86_64_linux/lib
    endif
    ifneq ($(TARGET_ARCH), aarch64)
        LIBS += -lEmulator -lvdtproxy
    endif
    LIBS += -lCLC -lGAL -lOpenVX -lOpenVXU -lVSC -lArchModelSw -lNNArchPerf
endif

ifeq ($(TARGET),ios)
	CORE_CC_EXCLUDE_SRCS += tensorflow/lite/minimal_logging_android.cc
	CORE_CC_EXCLUDE_SRCS += tensorflow/lite/minimal_logging_default.cc
else
	CORE_CC_EXCLUDE_SRCS += tensorflow/lite/minimal_logging_android.cc
	CORE_CC_EXCLUDE_SRCS += tensorflow/lite/minimal_logging_ios.cc
endif

# Filter out all the excluded files.
TF_LITE_CC_SRCS := $(filter-out $(CORE_CC_EXCLUDE_SRCS), $(CORE_CC_ALL_SRCS))

# Benchmark sources
BENCHMARK_SRCS_DIR := tensorflow/lite/tools/benchmark
DELEGATE_PROVIDER_SRCS_DIR := tensorflow/lite/tools/delegates
EVALUATION_UTILS_SRCS := \
  tensorflow/lite/tools/evaluation/utils.cc \

BENCHMARK_ALL_SRCS := $(filter-out \
	$(wildcard $(BENCHMARK_SRCS_DIR)/*_provider.cc), \
	$(wildcard $(BENCHMARK_SRCS_DIR)/*.cc)) \
	$(DELEGATE_PROVIDER_SRCS_DIR)/nnapi_delegate_provider.cc \
	$(DELEGATE_PROVIDER_SRCS_DIR)/default_execution_provider.cc \
	$(PROFILE_SUMMARIZER_SRCS) \
	$(CMD_LINE_TOOLS_SRCS) \
	$(EVALUATION_UTILS_SRCS)

BENCHMARK_MAIN_SRC := $(BENCHMARK_SRCS_DIR)/benchmark_main.cc
BENCHMARK_PERF_OPTIONS_SRC := \
	$(BENCHMARK_SRCS_DIR)/benchmark_tflite_performance_options_main.cc
BENCHMARK_LIB_SRCS := $(filter-out \
	$(wildcard $(BENCHMARK_SRCS_DIR)/*_test.cc) \
	$(BENCHMARK_MAIN_SRC) \
	$(BENCHMARK_PERF_OPTIONS_SRC) \
	$(BENCHMARK_SRCS_DIR)/benchmark_plus_flex_main.cc \
	$(DELEGATE_PROVIDER_SRCS_DIR)/external_delegate_provider.cc \
	$(DELEGATE_PROVIDER_SRCS_DIR)/gpu_delegate_provider.cc \
	$(DELEGATE_PROVIDER_SRCS_DIR)/hexagon_delegate_provider.cc \
	$(DELEGATE_PROVIDER_SRCS_DIR)/xnnpack_delegate_provider.cc, \
	$(BENCHMARK_ALL_SRCS))

EVALUATION_BINARY_SRCS := \
	tensorflow/core/platform/default/logging.cc \
	tensorflow/core/platform/default/env_time.cc \
	tensorflow/core/lib/jpeg/jpeg_mem.cc \
	tensorflow/core/lib/jpeg/jpeg_handle.cc \
	tensorflow/lite/tools/evaluation/utils.cc \
	tensorflow/lite/tools/evaluation/tasks/task_executor_main.cc \
	tensorflow/lite/tools/evaluation/tasks/task_executor.cc \
	tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.cc \
	tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.cc \
	tensorflow/lite/tools/evaluation/proto/preprocessing_steps.pb.cc \
	tensorflow/lite/tools/evaluation/stages/image_classification_stage.cc \
	tensorflow/lite/tools/evaluation/stages/object_detection_stage.cc \
	tensorflow/lite/tools/evaluation/stages/object_detection_average_precision_stage.cc \
	tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.cc \
	tensorflow/lite/tools/evaluation/stages/tflite_inference_stage.cc \
	tensorflow/lite/tools/evaluation/stages/topk_accuracy_eval_stage.cc \
	tensorflow/lite/tools/evaluation/stages/utils/image_metrics.cc \
	tensorflow/lite/tools/evaluation/stages/inference_profiler_stage.cc \
	tensorflow/lite/tools/evaluation/evaluation_delegate_provider.cc \
	tensorflow/lite/tools/make/downloads/absl/absl/hash/internal/hash.cc \
	tensorflow/lite/tools/make/downloads/absl/absl/hash/internal/city.cc \
	tensorflow/lite/tools/make/downloads/absl/absl/container/internal/hashtablez_sampler.cc \
	tensorflow/lite/tools/make/downloads/absl/absl/container/internal/hashtablez_sampler_force_weak_definition.cc \
	tensorflow/lite/tools/make/downloads/absl/absl/synchronization/mutex.cc \
	tensorflow/lite/tools/make/downloads/absl/absl/synchronization/internal/per_thread_sem.cc \
	tensorflow/lite/tools/make/downloads/absl/absl/synchronization/internal/create_thread_identity.cc \
	tensorflow/lite/tools/make/downloads/absl/absl/synchronization/internal/waiter.cc \
	tensorflow/lite/tools/make/downloads/absl/absl/debugging/symbolize.cc \
	tensorflow/lite/tools/make/downloads/absl/absl/debugging/stacktrace.cc \
	tensorflow/lite/tools/make/downloads/absl/absl/debugging/internal/demangle.cc \
	tensorflow/lite/tools/make/downloads/absl/absl/debugging/internal/vdso_support.cc \
	tensorflow/lite/tools/make/downloads/absl/absl/debugging/internal/elf_mem_image.cc \
	tensorflow/lite/tools/make/downloads/absl/absl/debugging/internal/address_is_readable.cc \
	$(CMD_LINE_TOOLS_SRCS) \
	$(DELEGATE_PROVIDER_SRCS_DIR)/nnapi_delegate_provider.cc \
	$(DELEGATE_PROVIDER_SRCS_DIR)/default_execution_provider.cc \

COCO_OBJECT_DETECTION_BINARY_NAME := coco_object_detection_run_eval
COCO_OBJECT_DETECTION_SRCS := \
	tensorflow/lite/tools/evaluation/tasks/coco_object_detection/run_eval.cc \
	$(EVALUATION_BINARY_SRCS)

IMAGENET_IMAGE_CLASSIFICATION_BINARY_NAME := imagenet_image_classification_run_eval
IMAGENET_IMAGE_CLASSIFICATION_SRCS := \
	tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification/run_eval.cc \
	$(EVALUATION_BINARY_SRCS)

INFERENCE_DIFF_BINARY_NAME := inference_diff_run_eval
INFERENCE_DIFF_SRCS := \
	tensorflow/lite/tools/evaluation/tasks/inference_diff/run_eval.cc \
	$(EVALUATION_BINARY_SRCS)

# These target-specific makefiles should modify or replace options like
# CXXFLAGS or LIBS to work for a specific targeted architecture. All logic
# based on platforms or architectures should happen within these files, to
# keep this main makefile focused on the sources and dependencies.
include $(wildcard $(MAKEFILE_DIR)/targets/*_makefile.inc)

ALL_SRCS := \
	$(MINIMAL_SRCS) \
	$(PROFILER_SRCS) \
	$(PROFILER_SUMMARIZER_SRCS) \
	$(TF_LITE_CC_SRCS) \
	$(BENCHMARK_LIB_SRCS) \
	$(CMD_LINE_TOOLS_SRCS)

# Where compiled objects are stored.
TARGET_OUT_DIR ?= $(TARGET)_$(TARGET_ARCH)
GENDIR := $(MAKEFILE_DIR)/gen/$(TARGET_OUT_DIR)/
OBJDIR := $(GENDIR)obj/
BINDIR := $(GENDIR)bin/
LIBDIR := $(GENDIR)lib/

LIB_PATH := $(LIBDIR)$(LIB_NAME)
BENCHMARK_LIB := $(LIBDIR)$(BENCHMARK_LIB_NAME)
BENCHMARK_BINARY := $(BINDIR)$(BENCHMARK_BINARY_NAME)
BENCHMARK_PERF_OPTIONS_BINARY := $(BINDIR)$(BENCHMARK_PERF_OPTIONS_BINARY_NAME)
MINIMAL_BINARY := $(BINDIR)minimal
LABEL_IMAGE_BINARY := $(BINDIR)label_image
COCO_OBJECT_DETECTION_BINARY := $(BINDIR)$(COCO_OBJECT_DETECTION_BINARY_NAME)
IMAGENET_IMAGE_CLASSIFICATION_BINARY := $(BINDIR)$(IMAGENET_IMAGE_CLASSIFICATION_BINARY_NAME)
INFERENCE_DIFF_BINARY := $(BINDIR)$(INFERENCE_DIFF_BINARY_NAME)


MINIMAL_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(MINIMAL_SRCS))))

LABEL_IMAGE_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(LABEL_IMAGE_SRCS))))

LIB_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(TF_LITE_CC_SRCS)))))

BENCHMARK_MAIN_OBJ := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(BENCHMARK_MAIN_SRC))))

BENCHMARK_PERF_OPTIONS_OBJ := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(BENCHMARK_PERF_OPTIONS_SRC))))

BENCHMARK_LIB_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(BENCHMARK_LIB_SRCS))))

UNIT_TEST_OBJS := $(addprefix $(OBJDIR), $(UNIT_TEST_SRCS:.cc=.o))
UNIT_TEST_UTIL_OBJS := $(addprefix $(OBJDIR), $(UNIT_TEST_UTIL_SRCS:.cc=.o))
UNIT_TEST_NAMES := $(addprefix $(OBJDIR), $(UNIT_TEST_SRCS:.cc=))

COCO_OBJECT_DETECTION_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(COCO_OBJECT_DETECTION_SRCS))))

IMAGENET_IMAGE_CLASSIFICATION_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(IMAGENET_IMAGE_CLASSIFICATION_SRCS))))

INFERENCE_DIFF_OBJS := $(addprefix $(OBJDIR), \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(INFERENCE_DIFF_SRCS))))

# For normal manually-created TensorFlow Lite C++ source files.
$(OBJDIR)%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# For normal manually-created TensorFlow Lite C source files.
$(OBJDIR)%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# The target that's compiled if there's no command-line arguments.
all: $(LIB_PATH)  $(MINIMAL_BINARY) $(BENCHMARK_BINARY) $(BENCHMARK_PERF_OPTIONS_BINARY) $(LABEL_IMAGE_BINARY) 

# The target that's compiled for micro-controllers
micro: $(LIB_PATH)

# Hack for generating schema file bypassing flatbuffer parsing
tensorflow/lite/schema/schema_generated.h:
	@cp -u tensorflow/lite/schema/schema_generated.h.OPENSOURCE tensorflow/lite/schema/schema_generated.h

# Gathers together all the objects we've compiled into a single '.a' archive.
$(LIB_PATH): tensorflow/lite/schema/schema_generated.h $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $(LIB_PATH) $(LIB_OBJS)

lib: $(LIB_PATH)

$(MINIMAL_BINARY): $(MINIMAL_OBJS) $(LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(MINIMAL_BINARY) $(MINIMAL_OBJS) \
	$(LIBFLAGS) $(LIB_PATH) $(LDFLAGS) $(LIBS)

minimal: $(MINIMAL_BINARY)

$(LABEL_IMAGE_BINARY): $(LABEL_IMAGE_OBJS) $(LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(LABEL_IMAGE_BINARY) $(LABEL_IMAGE_OBJS) \
	$(LIBFLAGS) $(LIB_PATH) $(LDFLAGS) $(LIBS)

label_image: $(LABEL_IMAGE_BINARY)

$(COCO_OBJECT_DETECTION_BINARY): $(COCO_OBJECT_DETECTION_OBJS) $(LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(COCO_OBJECT_DETECTION_BINARY) $(COCO_OBJECT_DETECTION_OBJS) \
	$(LIBFLAGS) $(LIB_PATH) $(LDFLAGS) $(LDOPTS) $(LIBS)

coco: $(COCO_OBJECT_DETECTION_BINARY)

$(IMAGENET_IMAGE_CLASSIFICATION_BINARY): $(IMAGENET_IMAGE_CLASSIFICATION_OBJS) $(LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(IMAGENET_IMAGE_CLASSIFICATION_BINARY) $(IMAGENET_IMAGE_CLASSIFICATION_OBJS) \
	$(LIBFLAGS) $(LIB_PATH) $(LDFLAGS) $(LDOPTS) $(LIBS)

imagenet: $(IMAGENET_IMAGE_CLASSIFICATION_BINARY)

$(INFERENCE_DIFF_BINARY): $(INFERENCE_DIFF_OBJS) $(LIB_PATH)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(INFERENCE_DIFF_BINARY) $(INFERENCE_DIFF_OBJS) \
	$(LIBFLAGS) $(LIB_PATH) $(LDFLAGS) $(LDOPTS) $(LIBS)

inference_diff: $(INFERENCE_DIFF_BINARY)

$(BENCHMARK_LIB) : $(LIB_PATH) $(BENCHMARK_LIB_OBJS)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $(BENCHMARK_LIB) $(LIB_OBJS) $(BENCHMARK_LIB_OBJS)

evaluation: coco imagenet inference_diff

benchmark_lib: $(BENCHMARK_LIB)

BENCHMARK_LINKOPTS :=
ifeq ($(HOST_OS),osx)
	BENCHMARK_LINKOPTS += $(LIBFLAGS) -Wl,-force_load $(BENCHMARK_LIB) $(LIBS) $(LDFLAGS) -framework CoreFoundation
else
	BENCHMARK_LINKOPTS += $(LIBFLAGS) -Wl,--whole-archive $(BENCHMARK_LIB) -Wl,--no-whole-archive $(LDFLAGS) $(LIBS)
endif

$(BENCHMARK_BINARY) : $(BENCHMARK_MAIN_OBJ) $(BENCHMARK_LIB)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(BENCHMARK_BINARY) $(BENCHMARK_MAIN_OBJ) \
	$(LIBFLAGS) $(BENCHMARK_LINKOPTS)

$(BENCHMARK_PERF_OPTIONS_BINARY) : $(BENCHMARK_PERF_OPTIONS_OBJ) $(BENCHMARK_LIB)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $(BENCHMARK_PERF_OPTIONS_BINARY) $(BENCHMARK_PERF_OPTIONS_OBJ) \
	$(LIBFLAGS) -Wl,--allow-multiple-definition -Wl,--whole-archive $(BENCHMARK_LIB) -Wl,--as-needed $(LDFLAGS) $(LIBS)

benchmark: $(BENCHMARK_BINARY) $(BENCHMARK_PERF_OPTIONS_BINARY)

tests: $(UNIT_TEST_NAMES)

LIB_DIRS := \
-L$(MAKEFILE_DIR)/downloads/googletest/build/lib/ \
-L$(MAKEFILE_DIR)/downloads/re2/build/ \
-L$(MAKEFILE_DIR)/downloads/absl/build/absl/algorithm \
-L$(MAKEFILE_DIR)/downloads/absl/build/absl/base \
-L$(MAKEFILE_DIR)/downloads/absl/build/absl/container \
-L$(MAKEFILE_DIR)/downloads/absl/build/absl/debugging \
-L$(MAKEFILE_DIR)/downloads/absl/build/absl/flags \
-L$(MAKEFILE_DIR)/downloads/absl/build/absl/hash \
-L$(MAKEFILE_DIR)/downloads/absl/build/absl/memory \
-L$(MAKEFILE_DIR)/downloads/absl/build/absl/meta \
-L$(MAKEFILE_DIR)/downloads/absl/build/absl/numeric \
-L$(MAKEFILE_DIR)/downloads/absl/build/absl/strings \
-L$(MAKEFILE_DIR)/downloads/absl/build/absl/synchronization \
-L$(MAKEFILE_DIR)/downloads/absl/build/absl/time \
-L$(MAKEFILE_DIR)/downloads/absl/build/absl/types \
-L$(MAKEFILE_DIR)/downloads/absl/build/absl/utility \
-L$(MAKEFILE_DIR)/downloads/absl/build/absl/status \



$(UNIT_TEST_NAMES): %: %.o $(UNIT_TEST_UTIL_OBJS) $(LIB_PATH)
	@mkdir -p $(dir $@) 
	$(CXX) $(CXXFLAGS) $(INCLUDES) \
	-o $@ $@.o $(UNIT_TEST_UTIL_OBJS) \
	$(LIB_PATH) $(LIBFLAGS) $(LDFLAGS) $(LIBS) \
	$(LIB_DIRS) \
	-lgtest -lgtest_main -lgmock -lre2 \
	-labsl_synchronization -labsl_stacktrace -labsl_scoped_set_env -labsl_raw_hash_set -labsl_debugging_internal -labsl_examine_stack -labsl_failure_signal_handler -labsl_leak_check -labsl_leak_check_disable -labsl_symbolize -labsl_demangle_internal -labsl_flags -labsl_flags_config -labsl_flags_internal -labsl_flags_marshalling -labsl_flags_parse -labsl_flags_registry -labsl_flags_usage -labsl_flags_usage_internal -labsl_flags_program_name -labsl_city -labsl_hash -labsl_hashtablez_sampler -labsl_str_format_internal -labsl_strings -labsl_strings_internal -labsl_graphcycles_internal -labsl_civil_time -labsl_time -labsl_time_zone -labsl_bad_any_cast_impl -labsl_bad_optional_access -labsl_bad_variant_access -labsl_base -labsl_malloc_internal -labsl_dynamic_annotations -labsl_int128 -labsl_throw_delegate -labsl_status -labsl_raw_logging_internal -labsl_log_severity -labsl_spinlock_wait -labsl_periodic_sampler -labsl_exponential_biased -labsl_cord 

libdir:
	@echo $(LIBDIR)

# Gets rid of all generated files.
clean:
	rm -rf $(MAKEFILE_DIR)/gen

# Gets rid of target files only, leaving the host alone. Also leaves the lib
# directory untouched deliberately, so we can persist multiple architectures
# across builds for iOS and Android.
cleantarget:
	rm -rf $(OBJDIR)
	rm -rf $(BINDIR)

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d

-include $(patsubst %,$(DEPDIR)/%.d,$(basename $(ALL_SRCS)))
