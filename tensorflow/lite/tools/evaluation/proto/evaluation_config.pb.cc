// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/lite/tools/evaluation/proto/evaluation_config.proto

#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
extern PROTOBUF_INTERNAL_EXPORT_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fstages_2eproto ::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<7> scc_info_ProcessMetrics_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fstages_2eproto;
extern PROTOBUF_INTERNAL_EXPORT_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fstages_2eproto ::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<6> scc_info_ProcessSpecification_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fstages_2eproto;
namespace tflite {
namespace evaluation {
class EvaluationStageConfigDefaultTypeInternal {
 public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<EvaluationStageConfig> _instance;
} _EvaluationStageConfig_default_instance_;
class EvaluationStageMetricsDefaultTypeInternal {
 public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<EvaluationStageMetrics> _instance;
} _EvaluationStageMetrics_default_instance_;
}  // namespace evaluation
}  // namespace tflite
static void InitDefaultsscc_info_EvaluationStageConfig_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fconfig_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::tflite::evaluation::_EvaluationStageConfig_default_instance_;
    new (ptr) ::tflite::evaluation::EvaluationStageConfig();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
  ::tflite::evaluation::EvaluationStageConfig::InitAsDefaultInstance();
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<1> scc_info_EvaluationStageConfig_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fconfig_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 1, InitDefaultsscc_info_EvaluationStageConfig_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fconfig_2eproto}, {
      &scc_info_ProcessSpecification_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fstages_2eproto.base,}};

static void InitDefaultsscc_info_EvaluationStageMetrics_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fconfig_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::tflite::evaluation::_EvaluationStageMetrics_default_instance_;
    new (ptr) ::tflite::evaluation::EvaluationStageMetrics();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
  ::tflite::evaluation::EvaluationStageMetrics::InitAsDefaultInstance();
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<1> scc_info_EvaluationStageMetrics_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fconfig_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 1, InitDefaultsscc_info_EvaluationStageMetrics_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fconfig_2eproto}, {
      &scc_info_ProcessMetrics_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fstages_2eproto.base,}};

static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fconfig_2eproto[2];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fconfig_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fconfig_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fconfig_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::tflite::evaluation::EvaluationStageConfig, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::tflite::evaluation::EvaluationStageConfig, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::tflite::evaluation::EvaluationStageConfig, name_),
  PROTOBUF_FIELD_OFFSET(::tflite::evaluation::EvaluationStageConfig, specification_),
  0,
  1,
  PROTOBUF_FIELD_OFFSET(::tflite::evaluation::EvaluationStageMetrics, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::tflite::evaluation::EvaluationStageMetrics, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::tflite::evaluation::EvaluationStageMetrics, num_runs_),
  PROTOBUF_FIELD_OFFSET(::tflite::evaluation::EvaluationStageMetrics, process_metrics_),
  1,
  0,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 7, sizeof(::tflite::evaluation::EvaluationStageConfig)},
  { 9, 16, sizeof(::tflite::evaluation::EvaluationStageMetrics)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::tflite::evaluation::_EvaluationStageConfig_default_instance_),
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::tflite::evaluation::_EvaluationStageMetrics_default_instance_),
};

const char descriptor_table_protodef_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fconfig_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n>tensorflow/lite/tools/evaluation/proto"
  "/evaluation_config.proto\022\021tflite.evaluat"
  "ion\032>tensorflow/lite/tools/evaluation/pr"
  "oto/evaluation_stages.proto\"e\n\025Evaluatio"
  "nStageConfig\022\014\n\004name\030\001 \001(\t\022>\n\rspecificat"
  "ion\030\002 \001(\0132\'.tflite.evaluation.ProcessSpe"
  "cification\"f\n\026EvaluationStageMetrics\022\020\n\010"
  "num_runs\030\001 \001(\005\022:\n\017process_metrics\030\002 \001(\0132"
  "!.tflite.evaluation.ProcessMetricsB\030\n\021tf"
  "lite.evaluationP\001\370\001\001"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fconfig_2eproto_deps[1] = {
  &::descriptor_table_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fstages_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase*const descriptor_table_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fconfig_2eproto_sccs[2] = {
  &scc_info_EvaluationStageConfig_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fconfig_2eproto.base,
  &scc_info_EvaluationStageMetrics_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fconfig_2eproto.base,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fconfig_2eproto_once;
static bool descriptor_table_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fconfig_2eproto_initialized = false;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fconfig_2eproto = {
  &descriptor_table_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fconfig_2eproto_initialized, descriptor_table_protodef_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fconfig_2eproto, "tensorflow/lite/tools/evaluation/proto/evaluation_config.proto", 380,
  &descriptor_table_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fconfig_2eproto_once, descriptor_table_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fconfig_2eproto_sccs, descriptor_table_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fconfig_2eproto_deps, 2, 1,
  schemas, file_default_instances, TableStruct_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fconfig_2eproto::offsets,
  file_level_metadata_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fconfig_2eproto, 2, file_level_enum_descriptors_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fconfig_2eproto, file_level_service_descriptors_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fconfig_2eproto,
};

// Force running AddDescriptors() at dynamic initialization time.
static bool dynamic_init_dummy_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fconfig_2eproto = (  ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptors(&descriptor_table_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fconfig_2eproto), true);
namespace tflite {
namespace evaluation {

// ===================================================================

void EvaluationStageConfig::InitAsDefaultInstance() {
  ::tflite::evaluation::_EvaluationStageConfig_default_instance_._instance.get_mutable()->specification_ = const_cast< ::tflite::evaluation::ProcessSpecification*>(
      ::tflite::evaluation::ProcessSpecification::internal_default_instance());
}
class EvaluationStageConfig::_Internal {
 public:
  using HasBits = decltype(std::declval<EvaluationStageConfig>()._has_bits_);
  static void set_has_name(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
  static const ::tflite::evaluation::ProcessSpecification& specification(const EvaluationStageConfig* msg);
  static void set_has_specification(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
};

const ::tflite::evaluation::ProcessSpecification&
EvaluationStageConfig::_Internal::specification(const EvaluationStageConfig* msg) {
  return *msg->specification_;
}
void EvaluationStageConfig::unsafe_arena_set_allocated_specification(
    ::tflite::evaluation::ProcessSpecification* specification) {
  if (GetArenaNoVirtual() == nullptr) {
    delete specification_;
  }
  specification_ = specification;
  if (specification) {
    _has_bits_[0] |= 0x00000002u;
  } else {
    _has_bits_[0] &= ~0x00000002u;
  }
  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:tflite.evaluation.EvaluationStageConfig.specification)
}
void EvaluationStageConfig::clear_specification() {
  if (specification_ != nullptr) specification_->Clear();
  _has_bits_[0] &= ~0x00000002u;
}
EvaluationStageConfig::EvaluationStageConfig()
  : ::PROTOBUF_NAMESPACE_ID::Message(), _internal_metadata_(nullptr) {
  SharedCtor();
  // @@protoc_insertion_point(constructor:tflite.evaluation.EvaluationStageConfig)
}
EvaluationStageConfig::EvaluationStageConfig(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
  _internal_metadata_(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:tflite.evaluation.EvaluationStageConfig)
}
EvaluationStageConfig::EvaluationStageConfig(const EvaluationStageConfig& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _internal_metadata_(nullptr),
      _has_bits_(from._has_bits_) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  name_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (from.has_name()) {
    name_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), from.name(),
      GetArenaNoVirtual());
  }
  if (from.has_specification()) {
    specification_ = new ::tflite::evaluation::ProcessSpecification(*from.specification_);
  } else {
    specification_ = nullptr;
  }
  // @@protoc_insertion_point(copy_constructor:tflite.evaluation.EvaluationStageConfig)
}

void EvaluationStageConfig::SharedCtor() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&scc_info_EvaluationStageConfig_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fconfig_2eproto.base);
  name_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  specification_ = nullptr;
}

EvaluationStageConfig::~EvaluationStageConfig() {
  // @@protoc_insertion_point(destructor:tflite.evaluation.EvaluationStageConfig)
  SharedDtor();
}

void EvaluationStageConfig::SharedDtor() {
  GOOGLE_DCHECK(GetArenaNoVirtual() == nullptr);
  name_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (this != internal_default_instance()) delete specification_;
}

void EvaluationStageConfig::ArenaDtor(void* object) {
  EvaluationStageConfig* _this = reinterpret_cast< EvaluationStageConfig* >(object);
  (void)_this;
}
void EvaluationStageConfig::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void EvaluationStageConfig::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const EvaluationStageConfig& EvaluationStageConfig::default_instance() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_EvaluationStageConfig_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fconfig_2eproto.base);
  return *internal_default_instance();
}


void EvaluationStageConfig::Clear() {
// @@protoc_insertion_point(message_clear_start:tflite.evaluation.EvaluationStageConfig)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000003u) {
    if (cached_has_bits & 0x00000001u) {
      name_.ClearNonDefaultToEmpty();
    }
    if (cached_has_bits & 0x00000002u) {
      GOOGLE_DCHECK(specification_ != nullptr);
      specification_->Clear();
    }
  }
  _has_bits_.Clear();
  _internal_metadata_.Clear();
}

#if GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
const char* EvaluationStageConfig::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  ::PROTOBUF_NAMESPACE_ID::Arena* arena = GetArenaNoVirtual(); (void)arena;
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // optional string name = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10)) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParserUTF8Verify(mutable_name(), ptr, ctx, "tflite.evaluation.EvaluationStageConfig.name");
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional .tflite.evaluation.ProcessSpecification specification = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 18)) {
          ptr = ctx->ParseMessage(mutable_specification(), ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag & 7) == 4 || tag == 0) {
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag, &_internal_metadata_, ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  _has_bits_.Or(has_bits);
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}
#else  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
bool EvaluationStageConfig::MergePartialFromCodedStream(
    ::PROTOBUF_NAMESPACE_ID::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!PROTOBUF_PREDICT_TRUE(EXPRESSION)) goto failure
  ::PROTOBUF_NAMESPACE_ID::uint32 tag;
  // @@protoc_insertion_point(parse_start:tflite.evaluation.EvaluationStageConfig)
  for (;;) {
    ::std::pair<::PROTOBUF_NAMESPACE_ID::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // optional string name = 1;
      case 1: {
        if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (10 & 0xFF)) {
          DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadString(
                input, this->mutable_name()));
          ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::VerifyUTF8StringNamedField(
            this->name().data(), static_cast<int>(this->name().length()),
            ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::PARSE,
            "tflite.evaluation.EvaluationStageConfig.name");
        } else {
          goto handle_unusual;
        }
        break;
      }

      // optional .tflite.evaluation.ProcessSpecification specification = 2;
      case 2: {
        if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (18 & 0xFF)) {
          DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadMessage(
               input, mutable_specification()));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0) {
          goto success;
        }
        DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SkipField(
              input, tag, _internal_metadata_.mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:tflite.evaluation.EvaluationStageConfig)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:tflite.evaluation.EvaluationStageConfig)
  return false;
#undef DO_
}
#endif  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER

void EvaluationStageConfig::SerializeWithCachedSizes(
    ::PROTOBUF_NAMESPACE_ID::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:tflite.evaluation.EvaluationStageConfig)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional string name = 1;
  if (cached_has_bits & 0x00000001u) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::VerifyUTF8StringNamedField(
      this->name().data(), static_cast<int>(this->name().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SERIALIZE,
      "tflite.evaluation.EvaluationStageConfig.name");
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteStringMaybeAliased(
      1, this->name(), output);
  }

  // optional .tflite.evaluation.ProcessSpecification specification = 2;
  if (cached_has_bits & 0x00000002u) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteMessageMaybeToArray(
      2, _Internal::specification(this), output);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SerializeUnknownFields(
        _internal_metadata_.unknown_fields(), output);
  }
  // @@protoc_insertion_point(serialize_end:tflite.evaluation.EvaluationStageConfig)
}

::PROTOBUF_NAMESPACE_ID::uint8* EvaluationStageConfig::InternalSerializeWithCachedSizesToArray(
    ::PROTOBUF_NAMESPACE_ID::uint8* target) const {
  // @@protoc_insertion_point(serialize_to_array_start:tflite.evaluation.EvaluationStageConfig)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional string name = 1;
  if (cached_has_bits & 0x00000001u) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::VerifyUTF8StringNamedField(
      this->name().data(), static_cast<int>(this->name().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SERIALIZE,
      "tflite.evaluation.EvaluationStageConfig.name");
    target =
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteStringToArray(
        1, this->name(), target);
  }

  // optional .tflite.evaluation.ProcessSpecification specification = 2;
  if (cached_has_bits & 0x00000002u) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessageToArray(
        2, _Internal::specification(this), target);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields(), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:tflite.evaluation.EvaluationStageConfig)
  return target;
}

size_t EvaluationStageConfig::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:tflite.evaluation.EvaluationStageConfig)
  size_t total_size = 0;

  if (_internal_metadata_.have_unknown_fields()) {
    total_size +=
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::ComputeUnknownFieldsSize(
        _internal_metadata_.unknown_fields());
  }
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000003u) {
    // optional string name = 1;
    if (cached_has_bits & 0x00000001u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
          this->name());
    }

    // optional .tflite.evaluation.ProcessSpecification specification = 2;
    if (cached_has_bits & 0x00000002u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
          *specification_);
    }

  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void EvaluationStageConfig::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:tflite.evaluation.EvaluationStageConfig)
  GOOGLE_DCHECK_NE(&from, this);
  const EvaluationStageConfig* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<EvaluationStageConfig>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:tflite.evaluation.EvaluationStageConfig)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:tflite.evaluation.EvaluationStageConfig)
    MergeFrom(*source);
  }
}

void EvaluationStageConfig::MergeFrom(const EvaluationStageConfig& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:tflite.evaluation.EvaluationStageConfig)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x00000003u) {
    if (cached_has_bits & 0x00000001u) {
      set_name(from.name());
    }
    if (cached_has_bits & 0x00000002u) {
      mutable_specification()->::tflite::evaluation::ProcessSpecification::MergeFrom(from.specification());
    }
  }
}

void EvaluationStageConfig::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:tflite.evaluation.EvaluationStageConfig)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void EvaluationStageConfig::CopyFrom(const EvaluationStageConfig& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:tflite.evaluation.EvaluationStageConfig)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool EvaluationStageConfig::IsInitialized() const {
  if (has_specification()) {
    if (!this->specification_->IsInitialized()) return false;
  }
  return true;
}

void EvaluationStageConfig::InternalSwap(EvaluationStageConfig* other) {
  using std::swap;
  _internal_metadata_.Swap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  name_.Swap(&other->name_, &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
    GetArenaNoVirtual());
  swap(specification_, other->specification_);
}

::PROTOBUF_NAMESPACE_ID::Metadata EvaluationStageConfig::GetMetadata() const {
  return GetMetadataStatic();
}


// ===================================================================

void EvaluationStageMetrics::InitAsDefaultInstance() {
  ::tflite::evaluation::_EvaluationStageMetrics_default_instance_._instance.get_mutable()->process_metrics_ = const_cast< ::tflite::evaluation::ProcessMetrics*>(
      ::tflite::evaluation::ProcessMetrics::internal_default_instance());
}
class EvaluationStageMetrics::_Internal {
 public:
  using HasBits = decltype(std::declval<EvaluationStageMetrics>()._has_bits_);
  static void set_has_num_runs(HasBits* has_bits) {
    (*has_bits)[0] |= 2u;
  }
  static const ::tflite::evaluation::ProcessMetrics& process_metrics(const EvaluationStageMetrics* msg);
  static void set_has_process_metrics(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
};

const ::tflite::evaluation::ProcessMetrics&
EvaluationStageMetrics::_Internal::process_metrics(const EvaluationStageMetrics* msg) {
  return *msg->process_metrics_;
}
void EvaluationStageMetrics::unsafe_arena_set_allocated_process_metrics(
    ::tflite::evaluation::ProcessMetrics* process_metrics) {
  if (GetArenaNoVirtual() == nullptr) {
    delete process_metrics_;
  }
  process_metrics_ = process_metrics;
  if (process_metrics) {
    _has_bits_[0] |= 0x00000001u;
  } else {
    _has_bits_[0] &= ~0x00000001u;
  }
  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:tflite.evaluation.EvaluationStageMetrics.process_metrics)
}
void EvaluationStageMetrics::clear_process_metrics() {
  if (process_metrics_ != nullptr) process_metrics_->Clear();
  _has_bits_[0] &= ~0x00000001u;
}
EvaluationStageMetrics::EvaluationStageMetrics()
  : ::PROTOBUF_NAMESPACE_ID::Message(), _internal_metadata_(nullptr) {
  SharedCtor();
  // @@protoc_insertion_point(constructor:tflite.evaluation.EvaluationStageMetrics)
}
EvaluationStageMetrics::EvaluationStageMetrics(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
  _internal_metadata_(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:tflite.evaluation.EvaluationStageMetrics)
}
EvaluationStageMetrics::EvaluationStageMetrics(const EvaluationStageMetrics& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _internal_metadata_(nullptr),
      _has_bits_(from._has_bits_) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  if (from.has_process_metrics()) {
    process_metrics_ = new ::tflite::evaluation::ProcessMetrics(*from.process_metrics_);
  } else {
    process_metrics_ = nullptr;
  }
  num_runs_ = from.num_runs_;
  // @@protoc_insertion_point(copy_constructor:tflite.evaluation.EvaluationStageMetrics)
}

void EvaluationStageMetrics::SharedCtor() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&scc_info_EvaluationStageMetrics_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fconfig_2eproto.base);
  ::memset(&process_metrics_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&num_runs_) -
      reinterpret_cast<char*>(&process_metrics_)) + sizeof(num_runs_));
}

EvaluationStageMetrics::~EvaluationStageMetrics() {
  // @@protoc_insertion_point(destructor:tflite.evaluation.EvaluationStageMetrics)
  SharedDtor();
}

void EvaluationStageMetrics::SharedDtor() {
  GOOGLE_DCHECK(GetArenaNoVirtual() == nullptr);
  if (this != internal_default_instance()) delete process_metrics_;
}

void EvaluationStageMetrics::ArenaDtor(void* object) {
  EvaluationStageMetrics* _this = reinterpret_cast< EvaluationStageMetrics* >(object);
  (void)_this;
}
void EvaluationStageMetrics::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void EvaluationStageMetrics::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const EvaluationStageMetrics& EvaluationStageMetrics::default_instance() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_EvaluationStageMetrics_tensorflow_2flite_2ftools_2fevaluation_2fproto_2fevaluation_5fconfig_2eproto.base);
  return *internal_default_instance();
}


void EvaluationStageMetrics::Clear() {
// @@protoc_insertion_point(message_clear_start:tflite.evaluation.EvaluationStageMetrics)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000001u) {
    GOOGLE_DCHECK(process_metrics_ != nullptr);
    process_metrics_->Clear();
  }
  num_runs_ = 0;
  _has_bits_.Clear();
  _internal_metadata_.Clear();
}

#if GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
const char* EvaluationStageMetrics::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  ::PROTOBUF_NAMESPACE_ID::Arena* arena = GetArenaNoVirtual(); (void)arena;
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // optional int32 num_runs = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 8)) {
          _Internal::set_has_num_runs(&has_bits);
          num_runs_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // optional .tflite.evaluation.ProcessMetrics process_metrics = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 18)) {
          ptr = ctx->ParseMessage(mutable_process_metrics(), ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag & 7) == 4 || tag == 0) {
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag, &_internal_metadata_, ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  _has_bits_.Or(has_bits);
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}
#else  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
bool EvaluationStageMetrics::MergePartialFromCodedStream(
    ::PROTOBUF_NAMESPACE_ID::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!PROTOBUF_PREDICT_TRUE(EXPRESSION)) goto failure
  ::PROTOBUF_NAMESPACE_ID::uint32 tag;
  // @@protoc_insertion_point(parse_start:tflite.evaluation.EvaluationStageMetrics)
  for (;;) {
    ::std::pair<::PROTOBUF_NAMESPACE_ID::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // optional int32 num_runs = 1;
      case 1: {
        if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (8 & 0xFF)) {
          _Internal::set_has_num_runs(&_has_bits_);
          DO_((::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadPrimitive<
                   ::PROTOBUF_NAMESPACE_ID::int32, ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_INT32>(
                 input, &num_runs_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // optional .tflite.evaluation.ProcessMetrics process_metrics = 2;
      case 2: {
        if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (18 & 0xFF)) {
          DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadMessage(
               input, mutable_process_metrics()));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0) {
          goto success;
        }
        DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SkipField(
              input, tag, _internal_metadata_.mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:tflite.evaluation.EvaluationStageMetrics)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:tflite.evaluation.EvaluationStageMetrics)
  return false;
#undef DO_
}
#endif  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER

void EvaluationStageMetrics::SerializeWithCachedSizes(
    ::PROTOBUF_NAMESPACE_ID::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:tflite.evaluation.EvaluationStageMetrics)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional int32 num_runs = 1;
  if (cached_has_bits & 0x00000002u) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32(1, this->num_runs(), output);
  }

  // optional .tflite.evaluation.ProcessMetrics process_metrics = 2;
  if (cached_has_bits & 0x00000001u) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteMessageMaybeToArray(
      2, _Internal::process_metrics(this), output);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SerializeUnknownFields(
        _internal_metadata_.unknown_fields(), output);
  }
  // @@protoc_insertion_point(serialize_end:tflite.evaluation.EvaluationStageMetrics)
}

::PROTOBUF_NAMESPACE_ID::uint8* EvaluationStageMetrics::InternalSerializeWithCachedSizesToArray(
    ::PROTOBUF_NAMESPACE_ID::uint8* target) const {
  // @@protoc_insertion_point(serialize_to_array_start:tflite.evaluation.EvaluationStageMetrics)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional int32 num_runs = 1;
  if (cached_has_bits & 0x00000002u) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(1, this->num_runs(), target);
  }

  // optional .tflite.evaluation.ProcessMetrics process_metrics = 2;
  if (cached_has_bits & 0x00000001u) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessageToArray(
        2, _Internal::process_metrics(this), target);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields(), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:tflite.evaluation.EvaluationStageMetrics)
  return target;
}

size_t EvaluationStageMetrics::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:tflite.evaluation.EvaluationStageMetrics)
  size_t total_size = 0;

  if (_internal_metadata_.have_unknown_fields()) {
    total_size +=
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::ComputeUnknownFieldsSize(
        _internal_metadata_.unknown_fields());
  }
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000003u) {
    // optional .tflite.evaluation.ProcessMetrics process_metrics = 2;
    if (cached_has_bits & 0x00000001u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
          *process_metrics_);
    }

    // optional int32 num_runs = 1;
    if (cached_has_bits & 0x00000002u) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
          this->num_runs());
    }

  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void EvaluationStageMetrics::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:tflite.evaluation.EvaluationStageMetrics)
  GOOGLE_DCHECK_NE(&from, this);
  const EvaluationStageMetrics* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<EvaluationStageMetrics>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:tflite.evaluation.EvaluationStageMetrics)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:tflite.evaluation.EvaluationStageMetrics)
    MergeFrom(*source);
  }
}

void EvaluationStageMetrics::MergeFrom(const EvaluationStageMetrics& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:tflite.evaluation.EvaluationStageMetrics)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = from._has_bits_[0];
  if (cached_has_bits & 0x00000003u) {
    if (cached_has_bits & 0x00000001u) {
      mutable_process_metrics()->::tflite::evaluation::ProcessMetrics::MergeFrom(from.process_metrics());
    }
    if (cached_has_bits & 0x00000002u) {
      num_runs_ = from.num_runs_;
    }
    _has_bits_[0] |= cached_has_bits;
  }
}

void EvaluationStageMetrics::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:tflite.evaluation.EvaluationStageMetrics)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void EvaluationStageMetrics::CopyFrom(const EvaluationStageMetrics& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:tflite.evaluation.EvaluationStageMetrics)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool EvaluationStageMetrics::IsInitialized() const {
  return true;
}

void EvaluationStageMetrics::InternalSwap(EvaluationStageMetrics* other) {
  using std::swap;
  _internal_metadata_.Swap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  swap(process_metrics_, other->process_metrics_);
  swap(num_runs_, other->num_runs_);
}

::PROTOBUF_NAMESPACE_ID::Metadata EvaluationStageMetrics::GetMetadata() const {
  return GetMetadataStatic();
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace evaluation
}  // namespace tflite
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::tflite::evaluation::EvaluationStageConfig* Arena::CreateMaybeMessage< ::tflite::evaluation::EvaluationStageConfig >(Arena* arena) {
  return Arena::CreateMessageInternal< ::tflite::evaluation::EvaluationStageConfig >(arena);
}
template<> PROTOBUF_NOINLINE ::tflite::evaluation::EvaluationStageMetrics* Arena::CreateMaybeMessage< ::tflite::evaluation::EvaluationStageMetrics >(Arena* arena) {
  return Arena::CreateMessageInternal< ::tflite::evaluation::EvaluationStageMetrics >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
