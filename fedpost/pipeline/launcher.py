from __future__ import annotations

import fedpost.algorithms
import fedpost.trainers
from fedpost.data.collators.sft_collator import SFTCollator
from fedpost.data.dataset_builder import DatasetBuilder
from fedpost.federation.client import Client
from fedpost.federation.coordinator import Coordinator
from fedpost.federation.sampler import UniformClientSampler
from fedpost.federation.server import Server
from fedpost.models.loader import HFModelManager
from fedpost.utils.recorder import Recorder
from fedpost.utils.registry import Registry


class Launcher:
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self):
        global_model_manager = HFModelManager(self.cfg)
        global_model_bundle = global_model_manager.build()

        dataset_builder = DatasetBuilder(self.cfg)
        fed_dataset = dataset_builder.build_federated_dataset()

        clients = self._build_clients(fed_dataset)

        algo_cls = Registry.get("algorithm", self.cfg.federated.algorithm)
        aggregator_cls = getattr(algo_cls, "aggregator_cls", None)
        if aggregator_cls is None:
            raise ValueError(
                f"Algorithm {self.cfg.federated.algorithm} does not define aggregator_cls"
            )
        aggregator = aggregator_cls(self.cfg)
        server = Server(
            cfg=self.cfg,
            global_model_manager=global_model_manager,
            model_bundle=global_model_bundle,
            aggregator=aggregator,
        )

        sampler = UniformClientSampler(self.cfg)
        algorithm = algo_cls(self.cfg, aggregator)

        evaluator = self._build_evaluator(global_model_bundle.tokenizer)
        recorder = Recorder(self.cfg.output_dir)

        coordinator = Coordinator(
            cfg=self.cfg,
            server=server,
            clients=clients,
            sampler=sampler,
            algorithm=algorithm,
            evaluator=evaluator,
            recorder=recorder,
        )
        return coordinator, recorder

    def run(self):
        coordinator, recorder = self.build()
        recorder.save_config(self.cfg)
        return coordinator.train()

    def _build_clients(self, fed_dataset):
        clients = []
        trainer_name = self.cfg.task
        if self.cfg.federated.algorithm == "bpfedpeft" and self.cfg.task == "sft":
            trainer_name = "bpfedpeft_sft"
        trainer_cls = Registry.get("trainer", trainer_name)

        for client_id in fed_dataset.get_client_ids():
            client_model_manager = HFModelManager(self.cfg)
            model_bundle = client_model_manager.build()
            tokenizer = model_bundle.tokenizer

            if self.cfg.task != "sft":
                raise ValueError("This anonymous BP-FedPEFT release keeps only the SFT path.")
            collator = SFTCollator(tokenizer, max_length=self.cfg.sft.max_length)

            trainer = trainer_cls(
                cfg=self.cfg,
                model_bundle=model_bundle,
                model_manager=client_model_manager,
                collator=collator,
            )

            client = Client(
                context=fed_dataset.get_client_context(client_id),
                trainer=trainer,
                dataset=fed_dataset.get_client_dataset(client_id),
            )
            clients.append(client)
        return clients

    def _build_evaluator(self, tokenizer):
        if not self.cfg.eval.tasks:
            return None
        from fedpost.evaluation.paper import PaperBenchmarkEvaluator

        return PaperBenchmarkEvaluator(self.cfg, tokenizer)
