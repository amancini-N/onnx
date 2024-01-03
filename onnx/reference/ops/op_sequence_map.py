# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


from onnx.reference.op_run import OpRun


class SequenceMap(OpRun):    
    def need_context(self) -> bool:
        return True

    def _run(self, input_sequence, *additional_inputs, body=None, attributes=None, context=None):  # type: ignore
        if len(additional_inputs) == 1 and isinstance(additional_inputs[0], list):
            res = None
            for obj1, obj2 in zip(input_sequence, additional_inputs[0]):
                feeds = {body.input_names[0]: obj1, body.input_names[1]: obj2}
                r = body.run(None, feeds)
                if res is None:
                    res = [[i] for i in r]
                else:
                    for s, i in zip(res, r):
                        s.append(i)
            return tuple(res)  # type: ignore

        feeds = dict(zip(body.input_names[1:], additional_inputs))
        if context:
            feeds.update(context)
        res = None
        for obj in input_sequence:
            feeds[body.input_names[0]] = obj
            r = body.run(None, feeds, attributes=attributes)
            if res is None:
                res = [[i] for i in r]
            else:
                for s, i in zip(res, r):
                    s.append(i)
        return tuple(res)  # type: ignore
