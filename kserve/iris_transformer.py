"""KServe Transformer for iris: friendly JSON in/out, V2 to the predictor."""
import argparse
import os
from typing import Any, Dict, List, Union

from kserve import InferInput, InferRequest, InferResponse, Model, ModelServer, model_server

IRIS_NAMES = ["setosa", "versicolor", "virginica"]


def _as_dict(response: Any) -> Dict:
    """Turn InferResponse / dict into a plain JSON-serializable dict."""
    if isinstance(response, dict):
        return response
    if isinstance(response, InferResponse):
        # Prefer REST-shaped dict when available
        if hasattr(response, "to_rest"):
            return response.to_rest()
        outputs = []
        for o in response.outputs:
            data = o.data
            if hasattr(data, "tolist"):
                data = data.tolist()
            outputs.append(
                {
                    "name": o.name,
                    "shape": list(o.shape) if o.shape is not None else None,
                    "datatype": o.datatype,
                    "data": data,
                }
            )
        return {
            "model_name": response.model_name,
            "id": getattr(response, "id", None),
            "outputs": outputs,
        }
    if hasattr(response, "dict"):
        return response.dict()
    return {"raw": str(response)}


def _class_ids(out: Dict) -> List[int]:
    outputs = out.get("outputs") or []
    if not outputs:
        # v1 style
        preds = out.get("predictions")
        if isinstance(preds, list):
            return [int(x) for x in preds]
        return []
    data = outputs[0].get("data") or []
    # flatten nested lists from some servers
    flat: List[int] = []
    for x in data:
        if isinstance(x, list):
            flat.extend(int(i) for i in x)
        else:
            flat.append(int(x))
    return flat


class IrisTransformer(Model):
    def __init__(self, name: str, predictor_host: str, protocol: str = "v2"):
        super().__init__(name)
        self.name = name
        self.predictor_host = predictor_host
        self.protocol = protocol
        self.ready = True

    async def preprocess(
        self, payload: Union[Dict, InferRequest], headers: Dict[str, str] = None
    ) -> Union[Dict, InferRequest]:
        """Accept {features|instances}; emit V2 InferRequest for sklearn predictor."""
        if isinstance(payload, InferRequest):
            return payload

        if "features" in payload:
            feats = payload["features"]
        elif "instances" in payload:
            feats = payload["instances"]
        else:
            return payload

        # Flatten row-major for InferInput
        rows = feats
        shape = [len(rows), len(rows[0])]
        flat = [float(v) for row in rows for v in row]
        infer_input = InferInput(
            name="input-0",
            shape=shape,
            datatype="FP32",
            data=flat,
        )
        return InferRequest(model_name=self.name, infer_inputs=[infer_input])

    async def postprocess(
        self, response: Union[Dict, InferResponse, list], headers: Dict[str, str] = None
    ) -> Dict:
        """Return plain dict (required) + human-readable labels."""
        out = _as_dict(response)
        ids = _class_ids(out)
        out["labels"] = [
            IRIS_NAMES[i] if 0 <= i < len(IRIS_NAMES) else str(i) for i in ids
        ]
        # Keep a simple field handy for clients
        out["predictions"] = ids
        return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[model_server.parser])
    args, _ = parser.parse_known_args()
    host = args.predictor_host or os.environ.get("PREDICTOR_HOST")
    if not host:
        raise SystemExit("predictor_host / PREDICTOR_HOST is required")
    protocol = (
        getattr(args, "predictor_protocol", None)
        or getattr(args, "protocol", None)
        or "v2"
    )
    model = IrisTransformer(args.model_name, predictor_host=host, protocol=protocol)
    ModelServer().start([model])
