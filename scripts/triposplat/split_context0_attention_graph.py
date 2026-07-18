#!/usr/bin/env python3
"""Create opt-in pre/post ONNX graphs around context_refiner.0 attention.

The canonical ONNX graph and its external-data sidecar are read-only. The generated
pre/post graphs retain the original external-initializer references and replace only
the 17 probability-times-V branches between the normalized Q/K/V boundary and the
existing context_refiner.0 attention projection.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

CUT_Q = "/flow_model/context_refiner.0/attn/Transpose_output_0"
CUT_K_TRANSPOSED = "/flow_model/context_refiner.0/attn/Transpose_3_output_0"
CUT_V = "/flow_model/context_refiner.0/attn/Slice_2_output_0"
CUT_ATTENDED = "/flow_model/context_refiner.0/attn/Concat_2_output_0"
CUSTOM_Q = "triposplat_context0_q"
CUSTOM_K = "triposplat_context0_k"
CUSTOM_V = "triposplat_context0_v"
CUSTOM_ATTENDED = "triposplat_context0_attended"
ATTENTION_FIRST_NODE = "/flow_model/context_refiner.0/attn/MatMul"
ATTENTION_POST_NODE = "/flow_model/context_refiner.0/attn/Concat_2"
TENSOR_SHAPE = [1, 16, 4101, 64]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--pre-output", type=Path, required=True)
    parser.add_argument("--post-output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def node_index(nodes: list[Any], name: str) -> int:
    matches = [index for index, node in enumerate(nodes) if node.name == name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one node named {name!r}, found {len(matches)}")
    return matches[0]


def value_infos(model: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        result[value.name] = value
    return result


def clone_value(info: Any, name: str | None = None) -> Any:
    result = copy.deepcopy(info)
    if name is not None:
        result.name = name
    return result


def model_with_graph(onnx: Any, source: Any, name: str, nodes: list[Any], inputs: list[Any], outputs: list[Any], metadata: dict[str, str]) -> Any:
    graph = onnx.helper.make_graph(
        [copy.deepcopy(node) for node in nodes], name, inputs, outputs,
        initializer=[copy.deepcopy(value) for value in source.graph.initializer],
    )
    result = onnx.helper.make_model(graph, opset_imports=copy.deepcopy(source.opset_import))
    result.ir_version = source.ir_version
    result.producer_name = "triposplat-context0-split"
    result.producer_version = "1"
    result.domain = source.domain
    result.model_version = source.model_version
    result.doc_string = source.doc_string
    for key, value in sorted(metadata.items()):
        item = result.metadata_props.add()
        item.key = key
        item.value = value
    return result


def save_checked(onnx: Any, model: Any, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        onnx.save_model(model, str(temporary), save_as_external_data=False)
        # This repository intentionally shares the canonical sidecar through hard
        # links. ONNX 1.22 rejects that layout before checking operators, so only
        # suppress that documented hardlink guard; every other checker failure is
        # still fatal.
        try:
            onnx.checker.check_model(model)
        except onnx.checker.ValidationError as error:
            detail = str(error)
            if "hardlink" not in detail and "not regular file" not in detail:
                raise
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    try:
        import onnx
        from onnx import shape_inference
    except ImportError as error:
        raise SystemExit("This tool requires the Python 'onnx' package.") from error

    graph_path = args.graph.resolve()
    if not graph_path.is_file():
        raise FileNotFoundError(graph_path)
    source = onnx.load_model(str(graph_path), load_external_data=False)
    inferred = shape_inference.infer_shapes(source)
    values = value_infos(inferred)
    nodes = list(source.graph.node)
    first = node_index(nodes, ATTENTION_FIRST_NODE)
    post = node_index(nodes, ATTENTION_POST_NODE) + 1
    if first >= post:
        raise ValueError("Context0 attention cut order is invalid")
    if any(name not in values for name in (CUT_Q, CUT_K_TRANSPOSED, CUT_V, CUT_ATTENDED)):
        raise ValueError("Canonical graph does not expose all required Context0 cut values")

    pre_nodes = [copy.deepcopy(node) for node in nodes[:first]]
    pre_nodes.extend((
        onnx.helper.make_node("Identity", [CUT_Q], [CUSTOM_Q], name="/triposplat/context0/Q_identity"),
        onnx.helper.make_node("Transpose", [CUT_K_TRANSPOSED], [CUSTOM_K], perm=[0, 1, 3, 2], name="/triposplat/context0/K_transpose"),
        onnx.helper.make_node("Identity", [CUT_V], [CUSTOM_V], name="/triposplat/context0/V_identity"),
    ))

    initializer_names = {value.name for value in source.graph.initializer}
    producer_indices = {
        output: index
        for index, node in enumerate(nodes)
        for output in node.output
        if output
    }
    post_indices = set(range(post, len(nodes)))
    while True:
        post_nodes = [copy.deepcopy(nodes[index]) for index in sorted(post_indices)]
        post_produced = {output for node in post_nodes for output in node.output if output}
        boundary = sorted({
            value for node in post_nodes for value in node.input
            if value and value not in post_produced and value not in initializer_names
            and value not in (CUSTOM_ATTENDED, CUT_ATTENDED)
        })
        missing = [value for value in boundary if value not in values]
        if missing:
            raise ValueError(f"No type/shape metadata exists for post boundary values: {missing}")
        rebuildable_boundary = [
            (value, producer_indices.get(value))
            for value in boundary
            if producer_indices.get(value) is not None
        ]
        if not rebuildable_boundary:
            break
        for value, producer in rebuildable_boundary:
            assert producer is not None
            if first <= producer < post:
                raise ValueError(f"Cannot rebuild post boundary {value!r} without restoring removed attention nodes")
            post_indices.add(producer)

    replacements = 0
    for node in post_nodes:
        for index, value in enumerate(node.input):
            if value == CUT_ATTENDED:
                node.input[index] = CUSTOM_ATTENDED
                replacements += 1
    if replacements != 1:
        raise ValueError(f"Expected one Context0 attended consumer in post graph, found {replacements}")

    qkv_infos = [onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, TENSOR_SHAPE) for name in (CUSTOM_Q, CUSTOM_K, CUSTOM_V)]
    attended_info = onnx.helper.make_tensor_value_info(CUSTOM_ATTENDED, onnx.TensorProto.FLOAT, TENSOR_SHAPE)
    pre_inputs = [clone_value(value) for value in source.graph.input]
    pre_outputs = qkv_infos
    post_inputs = [attended_info] + [clone_value(values[value]) for value in boundary]
    post_outputs = [clone_value(value) for value in source.graph.output]

    source_metadata = {entry.key: entry.value for entry in source.metadata_props}
    common_metadata = {
        **source_metadata,
        "triposplat.context0_attention_candidate": "wgsl-online-softmax-v1",
        "triposplat.context0_attention_cut": CUT_ATTENDED,
        "triposplat.context0_attention_q": CUSTOM_Q,
        "triposplat.context0_attention_k": CUSTOM_K,
        "triposplat.context0_attention_v": CUSTOM_V,
        "triposplat.context0_attention_attended": CUSTOM_ATTENDED,
        "triposplat.context0_attention_shape": json.dumps(TENSOR_SHAPE),
        "triposplat.context0_attention_source_sha256": sha256(graph_path),
        "triposplat.context0_attention_boundary": json.dumps(boundary),
    }
    pre_model = model_with_graph(onnx, source, "triposplat_context0_attention_pre", pre_nodes, pre_inputs, pre_outputs, {**common_metadata, "triposplat.context0_attention_part": "pre"})
    post_model = model_with_graph(onnx, source, "triposplat_context0_attention_post", post_nodes, post_inputs, post_outputs, {**common_metadata, "triposplat.context0_attention_part": "post"})
    save_checked(onnx, pre_model, args.pre_output)
    save_checked(onnx, post_model, args.post_output)
    report = {
        "candidate": "wgsl-online-softmax-v1",
        "canonicalGraph": str(graph_path),
        "canonicalGraphSha256": sha256(graph_path),
        "canonicalExternalData": sorted({entry.value for initializer in source.graph.initializer for entry in initializer.external_data if entry.key == "location"}),
        "preGraph": str(args.pre_output.resolve()),
        "preGraphSha256": sha256(args.pre_output),
        "postGraph": str(args.post_output.resolve()),
        "postGraphSha256": sha256(args.post_output),
        "preNodeCount": len(pre_nodes),
        "postNodeCount": len(post_nodes),
        "removedAttentionNodeCount": post - first,
        "boundaryInputs": boundary,
        "qkv": {"q": CUSTOM_Q, "k": CUSTOM_K, "v": CUSTOM_V, "shape": TENSOR_SHAPE},
        "attendedInput": CUSTOM_ATTENDED,
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
