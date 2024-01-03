# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


from typing import Any, List

import numpy as np

from onnx.reference.op_run import OpRun


def _concat_from_sequence(seq: List[Any], axis: int, new_axis: int = 0) -> np.ndarray:
    if new_axis == 1:
        new_slice = tuple([slice(None)] * axis + [np.newaxis] + [Ellipsis])
        seq2 = [s.__getitem__(new_slice) for s in seq]
        res = np.concatenate(seq2, axis=axis)
    else:
        res = np.concatenate(seq, axis=axis)
    return res  # type: ignore


class ConcatFromSequence(OpRun):
    def _run(self, seq, axis=None, new_axis=None):  # type: ignore
        if seq is None:
            raise RuntimeError("A sequence cannot be null.")
        res = _concat_from_sequence(seq, axis, new_axis=new_axis)
        return (res,)
