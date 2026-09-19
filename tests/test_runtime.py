from __future__ import annotations

import pytest

from tacyolo.runtime.backend import TensorRTBackend, create_backend


def test_tensorrt_backend_is_stub() -> None:
    with pytest.raises(NotImplementedError, match="Jetson"):
        TensorRTBackend("model.engine")


def test_dummy_backend_auto_when_weights_missing() -> None:
    backend = create_backend(weights="definitely_missing_model.pt", backend="auto")
    assert backend.name == "dummy"
