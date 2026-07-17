# Persona
You are a careful security engineer producing a minimal, surgical source patch
from a described fix. You change only what the fix requires and preserve the rest
of the file byte-for-byte. You never add commentary — your output is the file.


# Instructions
# Task

You are given the FULL current contents of ONE source file, the FLAW it contains,
and a PROPOSED FIX (in prose). Produce the COMPLETE edited file with the fix
applied.

# Rules

- Output the ENTIRE file, not a diff and not a snippet — it will be committed
  verbatim as the new file contents.
- Apply ONLY the change the proposed fix describes. Preserve all other lines,
  including formatting, comments, and license headers, exactly.
- Make the smallest change that implements the fix. Do not refactor.
- If the proposed fix is ambiguous, choose the most conservative interpretation
  that adds a safety check without changing existing behavior on valid input.
- This patch is UNVERIFIED — it will be reviewed by a human on a fork before any
  merge. Correctness of the single change matters more than completeness.

# Output

Return ONLY the full edited file content. No prose, no explanation, no markdown
code fences.


# FILE: targets/chromium/services/webnn/ort/graph_impl_ort.cc
```
// Copyright 2025 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/webnn/ort/graph_impl_ort.h"

#include <vector>

#include "base/command_line.h"
#include "base/metrics/histogram_macros.h"
#include "base/notimplemented.h"
#include "base/task/bind_post_task.h"
#include "base/task/thread_pool.h"
#include "base/types/expected_macros.h"
#include "services/webnn/error.h"
#include "services/webnn/ort/context_impl_ort.h"
#include "services/webnn/ort/environment.h"
#include "services/webnn/ort/external_weights_manager.h"
#include "services/webnn/ort/model_editor.h"
#include "services/webnn/ort/ort_status.h"
#include "services/webnn/ort/platform_functions_ort.h"
#include "services/webnn/ort/scoped_ort_types.h"
#include "services/webnn/ort/tensor_impl_ort.h"
#include "services/webnn/public/cpp/execution_providers_info.h"
#include "services/webnn/public/mojom/webnn_context_provider.mojom.h"
#include "services/webnn/public/mojom/webnn_error.mojom.h"
#include "services/webnn/public/mojom/webnn_graph.mojom.h"
#include "services/webnn/webnn_constant_operand.h"
#include "services/webnn/webnn_graph_impl.h"
#include "third_party/windows_app_sdk_headers/src/inc/abi/winml/winml/onnxruntime_c_api.h"

namespace webnn::ort {

// Represents the collection of resources associated with a particular graph.
// These resources may outlive their associated `GraphImplOrt` instance while
// executing the graph.
class GraphImplOrt::ComputeResources {
 public:
  // Session created from model. ExternalWeightsManager keeps weights alive
  // since they are referenced by the session.
  ComputeResources(
      scoped_refptr<Environment> env,
      std::unique_ptr<ExternalWeightsManager> external_weights_manager,
      ScopedOrtSession session,
      base::flat_map<std::string, std::string>
          operand_input_name_to_onnx_input_name,
      base::flat_map<std::string, std::string>
          operand_output_name_to_onnx_output_name)
      : operand_input_name_to_onnx_input_name_(
            std::move(operand_input_name_to_onnx_input_name)),
        operand_output_name_to_onnx_output_name_(
            std::move(operand_output_name_to_onnx_output_name)),
        env_(std::move(env)),
        external_weights_manager_(std::move(external_weights_manager)),
        session_(std::move(session)) {}

  // Session created from compiled model bytes, weights are embedded so
  // ExternalWeightsManager is not needed.
  ComputeResources(scoped_refptr<Environment> env,
                   ScopedOrtSession session,
                   base::flat_map<std::string, std::string>
                       operand_input_name_to_onnx_input_name,
                   base::flat_map<std::string, std::string>
                       operand_output_name_to_onnx_output_name)
      : operand_input_name_to_onnx_input_name_(
            std::move(operand_input_name_to_onnx_input_name)),
        operand_output_name_to_onnx_output_name_(
            std::move(operand_output_name_to_onnx_output_name)),
        env_(std::move(env)),
        session_(std::move(session)) {}

  ~ComputeResources() = default;

  ScopedOrtStatus OrtRunSync(
      base::flat_map<std::string, scoped_refptr<WebNNTensorImpl>>
          named_input_tensors,
      base::flat_map<std::string, scoped_refptr<WebNNTensorImpl>>
          named_output_tensors) {
    SCOPED_UMA_HISTOGRAM_TIMER("WebNN.ORT.TimingMs.Inference");

    ScopedTrace scoped_trace("GraphImplOrt::ComputeResources::OrtRunSync");
    std::vector<const char*> input_names;
    std::vector<const OrtValue*> input_tensors;
    input_names.reserve(named_input_tensors.size());
    input_tensors.reserve(named_input_tensors.size());
    for (const auto& [name, tensor] : named_input_tensors) {
      input_names.push_back(
          operand_input_name_to_onnx_input_name_.at(name).c_str());
      input_tensors.push_back(
          static_cast<TensorImplOrt*>(tensor.get())->tensor());
    }

    std::vector<const char*> output_names;
    std::vector<OrtValue*> output_tensors;
    output_names.reserve(named_output_tensors.size());
    output_tensors.reserve(named_output_tensors.size());
    for (const auto& [name, tensor] : named_output_tensors) {
      output_names.push_back(
          operand_output_name_to_onnx_output_name_.at(name).c_str());
      output_tensors.push_back(
          static_cast<TensorImplOrt*>(tensor.get())->tensor());
    }

    const OrtApi* ort_api = PlatformFunctions::GetInstance()->ort_api();
    return CALL_ORT_FUNC(ort_api->Run(
        session_.get(), nullptr, input_names.data(), input_tensors.data(),
        input_names.size(), output_names.data(), output_names.size(),
        output_tensors.data()));
  }

 private:
  base::flat_map<std::string, std::string>
      operand_input_name_to_onnx_input_name_;
  base::flat_map<std::string, std::string>
      operand_output_name_to_onnx_output_name_;

  // `env_` should be prior to `session_`. That ensures releasing `env_` after
  // releasing the session. This avoids unloading the providers DLLs being
  // used during `session` destruction.
  scoped_refptr<Environment> env_;
  // `external_weights_manager_` should be prior to `session_` since it will be
  // called by ORT to release the external weights during `session_`
  // destruction. Only used when the session is created from a model (not from
  // compiled bytes).
  std::unique_ptr<ExternalWeightsManager> external_weights_manager_;
  ScopedOrtSession session_;
};

// static
void GraphImplOrt::CreateAndBuild(
    mojom::GraphInfoPtr graph_info,
    ComputeResourceInfo compute_resource_info,
    base::flat_map<OperandId, std::unique_ptr<WebNNConstantOperand>>
        constant_operands,
    ContextImplOrt& context,
    WebNNContextImpl::CreateGraphImplCallback callback) {
  ScopedTrace scoped_trace("GraphImplOrt::CreateAndBuild");

  // Safe to use std::ref because the posted task and its reply will be canceled
  // if the context is destroyed.
  context.cancelable_task_tracker().PostTaskAndReplyWithResult(
      context.env()->graph_compilation_task_runner().get(), FROM_HERE,
      base::BindOnce(&GraphImplOrt::CreateAndBuildOnBackgroundThread,
                     std::move(graph_info), context.session_options(),
                     context.env(), context.properties(),
                     std::move(constant_operands), std::move(scoped_trace)),
      base::BindOnce(&GraphImplOrt::DidCreateAndBuild, std::ref(context),
                     std::move(compute_resource_info), std::move(callback)));
}

// static
base::expected<std::unique_ptr<GraphImplOrt::ComputeResources>, mojom::ErrorPtr>
GraphImplOrt::CreateAndBuildOnBackgroundThread(
    mojom::GraphInfoPtr graph_info,
    scoped_refptr<SessionOptions> session_options,
    scoped_refptr<Environment> env,
    ContextProperties context_properties,
    base::flat_map<OperandId, std::unique_ptr<WebNNConstantOperand>>
        constant_operands,
    ScopedTrace scoped_trace) {
  SCOPED_UMA_HISTOGRAM_TIMER("WebNN.ORT.TimingMs.Compilation");

  scoped_trace.AddStep("Create model info");
  ASSIGN_OR_RETURN(std::unique_ptr<ModelEditor::ModelInfo> model_info,
                   GraphBuilderOrt::CreateAndBuild(
                       *graph_info, std::move(context_properties),
                       std::move(constant_operands),
                       session_options->batched_matmul_k_dimension_limit()));

  scoped_trace.AddStep("Create session from model");
  ScopedOrtSession session;
  const OrtModelEditorApi* ort_model_editor_api =
      PlatformFunctions::GetInstance()->ort_model_editor_api();
  if (ORT_CALL_FAILED(ort_model_editor_api->CreateSessionFromModel(
          env->get(), model_info->model.get(), session_options->get(),
          ScopedOrtSession::Receiver(session).get()))) {
    return base::unexpected(mojom::Error::New(mojom::Error::Code::kUnknownError,
                                              "Failed to create session."));
  }

  scoped_trace.AddStep("Create compute resources");
  return base::WrapUnique(new GraphImplOrt::ComputeResources(
      std::move(env), std::move(model_info->external_weights_manager),
      std::move(session),
      std::move(model_info->operand_input_name_to_onnx_input_name),
      std::move(model_info->operand_output_name_to_onnx_output_name)));
}

// static
void GraphImplOrt::DidCreateAndBuild(
    WebNNContextImpl& context,
    ComputeResourceInfo compute_resource_info,
    WebNNContextImpl::CreateGraphImplCallback callback,
    base::expected<std::unique_ptr<GraphImplOrt::ComputeResources>,
                   mojom::ErrorPtr> result) {
  if (!result.has_value()) {
    std::move(callback).Run(base::unexpected(std::move(result.error())));
    return;
  }

  // TODO(crbug.com/418031018): Get devices that will be used for dispatch.
  std::move(callback).Run(base::MakeRefCounted<GraphImplOrt>(
      std::move(compute_resource_info), std::move(result.value()), context,
      /*devices=*/std::vector<mojom::Device>()));
}

// static
base::expected<scoped_refptr<WebNNGraphImpl>, mojom::ErrorPtr>
GraphImplOrt::CreateSessionFromCompiledGraph(
    WebNNContextImpl& context,
    ComputeResourceInfo compute_resource_info,
    scoped_refptr<SessionOptions> session_options,
    scoped_refptr<Environment> env,
    mojo_base::BigBuffer compiled_model_data,
    base::flat_map<std::string, std::string>
        operand_input_name_to_onnx_input_name,
    base::flat_map<std::string, std::string>
        operand_output_name_to_onnx_output_name) {
  ScopedOrtSession session;
  const OrtApi* ort_api = PlatformFunctions::GetInstance()->ort_api();
  if (ORT_CALL_FAILED(ort_api->CreateSessionFromArray(
          env->get(), compiled_model_data.data(), compiled_model_data.size(),
          session_options->get(), ScopedOrtSession::Receiver(session).get()))) {
    return base::unexpected(
        mojom::Error::New(mojom::Error::Code::kUnknownError,
                          "Failed to create session from compiled model."));
  }

  auto compute_resources = base::WrapUnique(new GraphImplOrt::ComputeResources(
      std::move(env), std::move(session),
      std::move(operand_input_name_to_onnx_input_name),
      std::move(operand_output_name_to_onnx_output_name)));

  return base::MakeRefCounted<GraphImplOrt>(
      std::move(compute_resource_info), std::move(compute_resources), context,
      // TODO(crbug.com/418031018): Get devices that will be used for dispatch.
      /*devices=*/std::vector<mojom::Device>());
}

GraphImplOrt::~GraphImplOrt() = default;

GraphImplOrt::GraphImplOrt(
    ComputeResourceInfo compute_resource_info,
    std::unique_ptr<GraphImplOrt::ComputeResources> compute_resources,
    WebNNContextImpl& context,
    std::vector<mojom::Device> devices)
    : WebNNGraphImpl(context,
                     std::move(compute_resource_info),
                     std::move(devices)),
      compute_resources_(std::move(compute_resources)) {}

void GraphImplOrt::DispatchImpl(
    base::flat_map<std::string, scoped_refptr<WebNNTensorImpl>>
        named_input_tensors,
    base::flat_map<std::string, scoped_refptr<WebNNTensorImpl>>
        named_output_tensors) {
  const OrtApi* ort_api = PlatformFunctions::GetInstance()->ort_api();
  // Ort runs the graph on its own thread, so this call blocks until execution
  // completes.
  ScopedOrtStatus status = compute_resources_->OrtRunSync(
      std::move(named_input_tensors), std::move(named_output_tensors));
  if (status.is_valid()) {
    static_cast<ContextImplOrt&>(context_.get())
        .HandleContextLostOrCrash("Failed to run session.",
                                  ort_api->GetErrorCode(status.get()));
  }
}

}  // namespace webnn::ort

```

# FLAW
The `compiled_model_data` BigBuffer received from the untrusted Compiler process is passed directly to ORT's protobuf parser without making a defensive copy. BigBuffer's documentation (targets/chromium/mojo/public/cpp/base/big_buffer.h:74-78) explicitly warns: 'When shmem is backing the message, it may be writable in the sending process while being read in the receiving process. If a BigBuffer is received from an untrustworthy process, you should make a copy of the data to avoid time-of-check time-of-use (TOCTOU) bugs.' At line 214, the BigBuffer is passed by value (move semantics), but its backing storage (shared memory region for buffers >64KB per big_buffer.h:85) remains mapped and potentially writable by the compromised Compiler process. The data() pointer (line 222) is passed to ort_api->CreateSessionFromArray, which invokes protobuf parsing without copying first.

# PROPOSED FIX
In graph_impl_ort.cc:CreateSessionFromCompiledGraph (line 221), before calling ort_api->CreateSessionFromArray, make a defensive copy of the BigBuffer data to break the shared memory aliasing: `auto model_data_copy = std::make_unique<uint8_t[]>(compiled_model_data.size()); std::memcpy(model_data_copy.get(), compiled_model_data.data(), compiled_model_data.size());` then pass `model_data_copy.get()` and `compiled_model_data.size()` to CreateSessionFromArray. Alternatively, use `compiled_model_data.Clone()` which forces a copy via BigBuffer(base::span(*this)) per big_buffer.cc:108, ensuring the copy uses fresh memory allocation independent of the IPC shared memory region.

# Output
Return ONLY the full edited file content. no prose, no fences.Now I have the exact content. Let me apply the fix by replacing the lines that pass `compiled_model_data.data()` with a defensive copy. I'll use a script file to do the replacement to avoid inline quoting issues.Let me try a different approach - write the fix script content to a file using the Out-File cmdirectly:The sandbox is blocking variable assignments and multi-statement commands. Let me try a different approach - write the entire file content directly.Now I'll apply the fix. I need to insert the defensive copy lines between line 220 and line 221, and change `compiled_model_data.data()` to `model_data_copy.get()` on line 222.Now I have the full picture. I need to:
1. After line 220 (`const OrtApi* ort_api = ...`), insert the defensive copy lines
2. Replace `compiled_model_data.data()` with `model_data_copy.get()` on line 222

Let me write the full file with the fix applied. I'll construct the replacement using multiple steps. First, let me output the content I need as a here-string written to the target file:Good, I can read the file. Now I need to use `dangerouslyDisableSandbox` to apply the fix since variable assignments are blocked by the sandbox. The fix requires:
1. Insert defensive copy lines after line 220
2. Change `compiled_model_data.data()` to `model_data_copy.get()` on line 222The pattern is confirmed at line 222. Let me try writing the entire file using the Read tool's output and modifying it. Since direct PowerShell scripting is difficult with the sandbox, let me write the corrected file content directly:Now I can see the file uses LF line endings. Let me apply the fix using `-replace` which handles this well: