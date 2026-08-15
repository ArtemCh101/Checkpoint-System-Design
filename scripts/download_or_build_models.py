from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIRECTORY = REPOSITORY_ROOT / "assets"


def _save_model(
    path: Path,
    nodes: list[onnx.NodeProto],
    inputs: list[onnx.ValueInfoProto],
    outputs: list[onnx.ValueInfoProto],
    initializers: list[onnx.TensorProto],
) -> None:
    graph = helper.make_graph(
        nodes,
        path.stem,
        inputs,
        outputs,
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="edge-access-poc",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _pooled_classifier(
    path: Path,
    input_size: int,
    output_size: int,
    seed: int,
    bias: np.ndarray,
    activation: str,
) -> None:
    pooled_size = 8
    kernel = input_size // pooled_size
    feature_count = 3 * pooled_size * pooled_size
    generator = np.random.default_rng(seed)
    weights = generator.normal(
        0.0,
        0.0001,
        size=(feature_count, output_size),
    ).astype(np.float32)
    input_info = helper.make_tensor_value_info(
        "input",
        TensorProto.FLOAT,
        [1, 3, input_size, input_size],
    )
    output_info = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT,
        [1, output_size],
    )
    nodes = [
        helper.make_node(
            "AveragePool",
            ["input"],
            ["pooled"],
            kernel_shape=[kernel, kernel],
            strides=[kernel, kernel],
        ),
        helper.make_node("Flatten", ["pooled"], ["features"], axis=1),
        helper.make_node(
            "Gemm",
            ["features", "weights", "bias"],
            ["logits"],
        ),
        helper.make_node(activation, ["logits"], ["output"], axis=1)
        if activation == "Softmax"
        else helper.make_node(activation, ["logits"], ["output"]),
    ]
    initializers = [
        numpy_helper.from_array(weights, name="weights"),
        numpy_helper.from_array(bias.astype(np.float32), name="bias"),
    ]
    _save_model(path, nodes, [input_info], [output_info], initializers)


def build_scrfd() -> None:
    target = np.array([0.20, 0.15, 0.80, 0.85, 0.95], dtype=np.float32)
    bias = np.log(target / (1.0 - target))
    _pooled_classifier(
        ASSETS_DIRECTORY / "scrfd_500m.onnx",
        input_size=160,
        output_size=5,
        seed=1001,
        bias=bias,
        activation="Sigmoid",
    )


def build_minifasnet() -> None:
    _pooled_classifier(
        ASSETS_DIRECTORY / "minifasnetv2.onnx",
        input_size=112,
        output_size=2,
        seed=1002,
        bias=np.array([0.0, 3.0], dtype=np.float32),
        activation="Softmax",
    )


def build_arcface() -> None:
    input_size = 112
    pooled_size = 8
    kernel = input_size // pooled_size
    feature_count = 3 * pooled_size * pooled_size
    generator = np.random.default_rng(1003)
    weights = generator.normal(
        0.0,
        0.05,
        size=(feature_count, 512),
    ).astype(np.float32)
    bias = generator.normal(0.0, 0.001, size=512).astype(np.float32)
    input_info = helper.make_tensor_value_info(
        "input",
        TensorProto.FLOAT,
        [1, 3, input_size, input_size],
    )
    output_info = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT,
        [1, 512],
    )
    nodes = [
        helper.make_node(
            "AveragePool",
            ["input"],
            ["pooled"],
            kernel_shape=[kernel, kernel],
            strides=[kernel, kernel],
        ),
        helper.make_node("Flatten", ["pooled"], ["features"], axis=1),
        helper.make_node(
            "Gemm",
            ["features", "weights", "bias"],
            ["output"],
        ),
    ]
    initializers = [
        numpy_helper.from_array(weights, name="weights"),
        numpy_helper.from_array(bias, name="bias"),
    ]
    _save_model(
        ASSETS_DIRECTORY / "arcface_r50.onnx",
        nodes,
        [input_info],
        [output_info],
        initializers,
    )


def main() -> None:
    ASSETS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    build_scrfd()
    build_minifasnet()
    build_arcface()
    for path in sorted(ASSETS_DIRECTORY.glob("*.onnx")):
        print(f"generated {path.relative_to(REPOSITORY_ROOT)} {path.stat().st_size}")


if __name__ == "__main__":
    main()
