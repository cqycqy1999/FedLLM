from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ClientContext:
    client_id: str
    num_samples: int
    metadata: dict = field(default_factory=dict)


class FederatedDataset:
    def __init__(self, client_to_data: dict[str, list], client_contexts: dict[str, ClientContext]):
        self.client_to_data = client_to_data
        self.client_contexts = client_contexts

    def get_client_ids(self) -> list[str]:
        return list(self.client_to_data.keys())

    def get_client_dataset(self, client_id: str) -> list:
        return self.client_to_data[client_id]

    def get_client_context(self, client_id: str) -> ClientContext:
        return self.client_contexts[client_id]

    def get_num_clients(self) -> int:
        return len(self.client_to_data)

    def summary(self) -> dict:
        return {
            "num_clients": self.get_num_clients(),
            "samples_per_client": {
                cid: len(data) for cid, data in self.client_to_data.items()
            },
        }