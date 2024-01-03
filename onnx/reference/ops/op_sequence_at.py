# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


from onnx.reference.op_run import OpRun


class SequenceAt(OpRun):
    def _run(self, seq, index):  # type: ignore
        if isinstance(index, np.ndarray):
            index = np.reshape(index, ())
        return (seq[index],)
