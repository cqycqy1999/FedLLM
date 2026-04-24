from __future__ import annotations

import math

import torch
from fedpost.algorithms.base import FederatedAlgorithm
from fedpost.utils.registry import Registry


class FedAvgAggregator:
    def __init__(self, cfg):
        self.cfg = cfg

    def aggregate(self, client_results):
        valid_results = [r for r in client_results if r.success]
        if not valid_results:
            raise RuntimeError("No valid client results to aggregate.")

        reference_result, update_schema = self._plan_aggregation(valid_results)
        total = sum(r.num_train_samples for r in valid_results)
        keys = tuple(reference_result.update.keys())

        aggregated = {}
        for key in keys:
            agg_value = None
            for r in valid_results:
                weight = r.num_train_samples / total
                value = r.update[key] * weight
                agg_value = value if agg_value is None else (agg_value + value)
            aggregated[key] = agg_value

        metrics = {
            "num_success_clients": len(valid_results),
            "aggregated_client_ids": [result.client_id for result in valid_results],
            "aggregation_reference_client_id": reference_result.client_id,
            "aggregation_mode": "homogeneous_strict",
            "aggregation_update_keys": list(keys),
            "aggregation_update_key_count": len(keys),
            "aggregation_update_schema": update_schema,
        }
        return aggregated, metrics

    # This strict compatibility gate is intentionally isolated so a future
    # heterogeneous PEFT aggregator can override it and group clients by shape.
    def _plan_aggregation(self, valid_results):
        reference_result = valid_results[0]
        reference_schema = self._collect_update_schema(reference_result)
        reference_keys = set(reference_schema)

        for result in valid_results[1:]:
            result_schema = self._collect_update_schema(result)
            result_keys = set(result_schema)

            missing_keys = sorted(reference_keys - result_keys)
            extra_keys = sorted(result_keys - reference_keys)
            if missing_keys or extra_keys:
                raise ValueError(
                    f"Client {result.client_id} update keys are incompatible with "
                    f"reference client {reference_result.client_id}. "
                    f"missing={missing_keys} extra={extra_keys}"
                )

            mismatches = []
            for key in reference_keys:
                ref_meta = reference_schema[key]
                result_meta = result_schema[key]
                for field_name in ("shape", "dtype", "device"):
                    if ref_meta[field_name] != result_meta[field_name]:
                        mismatches.append(
                            f"{key}.{field_name}: {ref_meta[field_name]} != {result_meta[field_name]}"
                        )

            if mismatches:
                raise ValueError(
                    f"Client {result.client_id} update tensors are incompatible with "
                    f"reference client {reference_result.client_id}: {'; '.join(mismatches[:10])}"
                )

        return reference_result, reference_schema

    def _collect_update_schema(self, result):
        schema = {}
        for key, value in result.update.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"Client {result.client_id} produced non-tensor update for key {key}: "
                    f"{type(value).__name__}"
                )

            if value.ndim == 0 and value.dtype.is_floating_point:
                scalar_value = float(value.detach().cpu().item())
                if math.isnan(scalar_value) or math.isinf(scalar_value):
                    raise ValueError(
                        f"Client {result.client_id} produced non-finite update for key {key}"
                    )
            elif value.dtype.is_floating_point or value.dtype.is_complex:
                if not torch.isfinite(value).all():
                    raise ValueError(
                        f"Client {result.client_id} produced non-finite update for key {key}"
                    )

            schema[key] = {
                "shape": tuple(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
        return schema


@Registry.register("algorithm", "fedavg")
class FedAvgAlgorithm(FederatedAlgorithm):
    pass
