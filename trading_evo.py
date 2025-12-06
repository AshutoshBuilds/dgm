"""
Evolutionary ProFiT-style loop with UCB/MCTS-inspired selection.
Uses a single local LLM to author full backtesting.py strategies (no seed template).
Supports host-side evaluation; container evaluation is available via Docker harness.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import random
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from llm import create_client, get_response_from_llm
from trading_backtest import evaluate_strategy, load_bars, save_stats
from trading_self_improve import extract_code, save_code
from utils import docker_utils


DEFAULT_MODEL = os.getenv("CODE_MODEL", "hf-local:hf_models/Qwen2.5-7B-Instruct")
DEFAULT_OUTDIR = Path("output_selfimprove_local_evo")


@dataclass
class StrategyNode:
    node_id: str
    parent_id: Optional[str]
    generation: int
    code_path: Path
    stats_path: Path
    fitness: float
    visits: int = 0
    children: List[str] = field(default_factory=list)
    ucb: float = 0.0
    novelty: float = 0.0
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvoConfig:
    pop_size: int = 20
    generations: int = 15
    ucb_c: float = 2.0
    prune_size: int = 40
    commission: float = 0.0005
    cash: float = 100_000.0
    container_eval: bool = False


def call_local_llm(prompt: str, system: str, model: str, temperature: float = 0.35) -> str:
    client, model_name = create_client(model)
    response, _ = get_response_from_llm(
        msg=prompt,
        client=client,
        model=model_name,
        system_message=system,
        print_debug=False,
        msg_history=None,
        temperature=temperature,
    )
    return response


def compute_fitness(stats: Dict[str, Any]) -> float:
    ann_return = float(stats.get("Return [%]", 0.0))
    drawdown = float(stats.get("Max. Drawdown [%]", 0.0))
    sharpe = float(stats.get("Sharpe Ratio", 0.0))
    return ann_return - max(drawdown, 0) * 0.5 + sharpe * 5


def ucb_score(node: StrategyNode, total_visits: int, c: float, novelty_weight: float = 0.5) -> float:
    exploitation = node.fitness
    exploration = c * math.sqrt(math.log(total_visits + 1) / (node.visits + 1))
    return exploitation + exploration + novelty_weight * node.novelty


def summarize_data(df: pd.DataFrame) -> str:
    start = df.index.min()
    end = df.index.max()
    rows = len(df)
    symbols = df["symbol"].unique().tolist() if "symbol" in df.columns else []
    has_oi = "open_interest" in df.columns
    return f"Rows={rows}, Start={start}, End={end}, Symbols={symbols}, HasOpenInterest={has_oi}"


def build_prompt(
    data_summary: str,
    parent_code: Optional[str],
    parent_stats: Optional[Dict[str, Any]],
) -> Tuple[str, str]:
    system = (
        "You are a quantitative trading researcher. "
        "Generate only valid Python code for backtesting.py. No explanations, no markdown—code only."
    )
    base_req = (
        "Create a complete class GeneratedStrategy(Strategy) using backtesting.py. "
        "Columns available: Open, High, Low, Close, Volume (index is timestamp). "
        "Avoid lookahead; use past-only signals. Keep code concise and vectorized. "
        "Include tunable class variables with sensible defaults. No external data/APIs."
    )
    if parent_code:
        prior = json.dumps(parent_stats) if parent_stats else "none"
        user = (
            f"{base_req}\n"
            f"Data summary: {data_summary}\n"
            f"Parent stats: {prior}\n"
            "Modify/improve the parent strategy below. Keep class name GeneratedStrategy.\n"
            "Parent code:\n"
            f"{parent_code}\n"
        )
    else:
        user = (
            f"{base_req}\n"
            f"Data summary: {data_summary}\n"
            "Design from scratch. Output only code."
        )
    return system, user


def evaluate_host(code: str, data: pd.DataFrame, cash: float, commission: float) -> Tuple[Dict[str, Any], float]:
    stats, fitness = evaluate_strategy(
        strategy_code=code,
        data=data,
        class_name="GeneratedStrategy",
        cash=cash,
        commission=commission,
    )
    return stats, fitness


def evaluate_in_container(
    code: str,
    data_path: Path,
    cash: float,
    commission: float,
    volumes: Dict[str, Dict[str, str]],
) -> Tuple[Dict[str, Any], float]:
    client = docker_utils.docker.DockerClient(base_url="npipe:////./pipe/docker_engine", timeout=120)
    container_name = f"dgm-evo-{uuid.uuid4().hex[:8]}"
    docker_utils.remove_existing_container(client, container_name)
    container = docker_utils.build_dgm_container(
        client,
        repo_path=".",
        image_name="dgm",
        container_name=container_name,
        force_rebuild=False,
        use_gpu=True,
        volumes=volumes,
    )
    if container is None:
        raise RuntimeError("Failed to start container for evaluation")

    try:
        # Write strategy code into container using tar archive
        archive = docker_utils.create_archive(Path("/tmp/strategy.py"), data=code.encode("utf-8"))
        container.put_archive("/", archive)

        cmd = [
            "python",
            "-c",
            (
                "import json, pandas as pd; "
                "from trading_backtest import evaluate_strategy, load_bars; "
                f"df = load_bars('{data_path.as_posix()}'); "
                "code=open('/tmp/strategy.py','r',encoding='utf-8').read(); "
                f"stats, fit = evaluate_strategy(code, df, class_name='GeneratedStrategy', cash={cash}, commission={commission}); "
                "print(json.dumps({'stats': stats, 'fitness': fit}))"
            ),
        ]
        exec_result = container.exec_run(cmd, workdir="/dgm")
        output = exec_result.output.decode("utf-8").strip()
        docker_utils.log_container_output(exec_result)
        payload = json.loads(output)
        return payload["stats"], float(payload["fitness"])
    finally:
        docker_utils.cleanup_container(container)


class EvolutionEngine:
    def __init__(
        self,
        data_path: Path,
        model: str,
        cfg: EvoConfig,
        outdir: Path = DEFAULT_OUTDIR,
        use_container: bool = False,
    ):
        self.data_path = data_path
        self.model = model
        self.cfg = cfg
        self.use_container = use_container
        self.data = load_bars(data_path)
        self.data_summary = summarize_data(self.data)
        self.run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = outdir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.nodes: Dict[str, StrategyNode] = {}
        self.root_ids: List[str] = []

    def _gen_node_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def _generate_code(self, parent_code: Optional[str], parent_stats: Optional[Dict[str, Any]]) -> str:
        system, prompt = build_prompt(self.data_summary, parent_code, parent_stats)
        raw = call_local_llm(prompt, system, self.model)
        return extract_code(raw)

    def _evaluate(self, code: str) -> Tuple[Dict[str, Any], float]:
        if self.use_container:
            # Mount data and hf_models into container
            volumes = {}
            data_dir = str(self.data_path.parent.resolve())
            volumes[data_dir] = {"bind": "/data", "mode": "ro"}
            if os.path.isdir("hf_models"):
                volumes[str(Path("hf_models").resolve())] = {"bind": "/dgm/hf_models", "mode": "ro"}
            # use the same relative path inside container
            container_data_path = Path("/data") / self.data_path.name
            return evaluate_in_container(
                code=code,
                data_path=container_data_path,
                cash=self.cfg.cash,
                commission=self.cfg.commission,
                volumes=volumes,
            )
        return evaluate_host(code, self.data, cash=self.cfg.cash, commission=self.cfg.commission)

    def _add_node(
        self,
        gen: int,
        parent_id: Optional[str],
        code: str,
        stats: Dict[str, Any],
        fitness: float,
    ) -> StrategyNode:
        node_id = self._gen_node_id()
        code_path = self.run_dir / f"{node_id}.py"
        stats_path = self.run_dir / f"{node_id}.json"
        save_code(code, code_path)
        save_stats(stats, stats_path)
        novelty = 1.0  # basic novelty bonus; can be extended with code hashing
        node = StrategyNode(
            node_id=node_id,
            parent_id=parent_id,
            generation=gen,
            code_path=code_path,
            stats_path=stats_path,
            fitness=fitness,
            novelty=novelty,
            stats=stats,
        )
        self.nodes[node_id] = node
        if parent_id:
            self.nodes[parent_id].children.append(node_id)
        else:
            self.root_ids.append(node_id)
        return node

    def _select_parent(self, population: List[StrategyNode]) -> StrategyNode:
        total_visits = sum(n.visits for n in population) + 1
        best = max(population, key=lambda n: ucb_score(n, total_visits, self.cfg.ucb_c))
        best.visits += 1
        return best

    def _prune(self, nodes: List[StrategyNode]) -> List[StrategyNode]:
        # Keep top by fitness up to prune_size
        nodes_sorted = sorted(nodes, key=lambda n: n.fitness, reverse=True)
        return nodes_sorted[: self.cfg.prune_size]

    def run(self):
        pop: List[StrategyNode] = []
        # Generation 0: fresh strategies
        for _ in range(self.cfg.pop_size):
            code = self._generate_code(parent_code=None, parent_stats=None)
            try:
                stats, fitness = self._evaluate(code)
            except Exception as exc:
                stats, fitness = {"error": str(exc)}, -math.inf
            node = self._add_node(gen=0, parent_id=None, code=code, stats=stats, fitness=fitness)
            pop.append(node)

        for gen in range(1, self.cfg.generations + 1):
            new_pop: List[StrategyNode] = []
            for _ in range(self.cfg.pop_size):
                parent = self._select_parent(pop)
                parent_code = Path(parent.code_path).read_text(encoding="utf-8")
                parent_stats = parent.stats
                code = self._generate_code(parent_code=parent_code, parent_stats=parent_stats)
                try:
                    stats, fitness = self._evaluate(code)
                except Exception as exc:
                    stats, fitness = {"error": str(exc)}, -math.inf
                child = self._add_node(gen=gen, parent_id=parent.node_id, code=code, stats=stats, fitness=fitness)
                new_pop.append(child)
            # Merge and prune
            combined = pop + new_pop
            pop = self._prune(combined)
        self._write_summary()

    def _write_summary(self):
        best = max(self.nodes.values(), key=lambda n: n.fitness if n.fitness is not None else -math.inf)
        lineage = {
            "run_id": self.run_id,
            "data": str(self.data_path),
            "model": self.model,
            "cfg": asdict(self.cfg),
            "nodes": [asdict(n) for n in self.nodes.values()],
            "best": asdict(best),
        }
        save_stats(lineage, self.run_dir / "lineage.json")
        print(f"Evolution run complete. Best fitness: {best.fitness}. Output: {self.run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evolutionary trading search with UCB/MCTS selection.")
    parser.add_argument("--data", type=Path, required=True, help="Path to processed parquet bars.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Local model id (hf-local:...).")
    parser.add_argument("--pop-size", type=int, default=20, help="Population size per generation.")
    parser.add_argument("--generations", type=int, default=15, help="Number of generations.")
    parser.add_argument("--ucb-c", type=float, default=2.0, help="UCB exploration constant.")
    parser.add_argument("--prune-size", type=int, default=40, help="Max nodes kept after pruning.")
    parser.add_argument("--cash", type=float, default=100_000.0, help="Starting cash.")
    parser.add_argument("--commission", type=float, default=0.0005, help="Per-trade commission fraction.")
    parser.add_argument("--container", action="store_true", help="Evaluate inside DGM container with GPU.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EvoConfig(
        pop_size=args.pop_size,
        generations=args.generations,
        ucb_c=args.ucb_c,
        prune_size=args.prune_size,
        commission=args.commission,
        cash=args.cash,
        container_eval=args.container,
    )
    engine = EvolutionEngine(
        data_path=args.data,
        model=args.model,
        cfg=cfg,
        use_container=args.container,
    )
    engine.run()


if __name__ == "__main__":
    main()

