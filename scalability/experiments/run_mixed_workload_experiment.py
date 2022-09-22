#!/usr/bin/env python
"""
Purpose: Measure IC performance give a complex workload.

The workload configuration to use is being read from a seperate workload description file.
"""
import json
import os
import shutil
import sys

import gflags
import toml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common.misc as misc  # noqa
import common.workload_experiment as workload_experiment  # noqa
import common.workload as workload  # noqa
import common.report as report  # noqa

FLAGS = gflags.FLAGS
gflags.DEFINE_string("workload", None, "Workload description to execute")
gflags.MarkFlagAsRequired("workload")

gflags.DEFINE_integer("initial_rps", 100, "Starting number for requests per second.")
gflags.DEFINE_integer("increment_rps", 50, "Increment of requests per second per round.")
gflags.DEFINE_integer(
    "max_rps", 40000, "Maximum requests per second to be sent. Experiment will wrap up beyond this number."
)

NUM_MACHINES_PER_WORKLOAD = 1  # TODO - make configurable in toml


class BytesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return obj.decode("utf-8")
        return json.JSONEncoder.default(self, obj)


class MixedWorkloadExperiment(workload_experiment.WorkloadExperiment):
    """Logic for mixed workload experiments."""

    def __init__(self):
        """Install canisters."""
        super().__init__()
        self.workload_description = None
        shutil.copy(FLAGS.workload, self.out_dir)
        with open(FLAGS.workload) as f:
            self.raw_description = toml.loads(f.read())
            self.install_canister_from_workload_description(self.raw_description)
            self.workload_description = workload.workload_description_from_dict(self.raw_description, self.canister_ids)

    def install_canister_from_workload_description(self, description):
        """Install all canisters required to run the given workload description."""
        for wl in description["workload"]:
            canister = wl["canister"]
            if canister not in self.canister_ids:
                self.install_canister(self.target_nodes[0], canister)

    def run_experiment_internal(self, config):
        """Run workload generator with the load specified in config."""
        f_stdout = os.path.join(self.iter_outdir, "workload-generator-{}.stdout.txt")
        f_stderr = os.path.join(self.iter_outdir, "workload-generator-{}.stderr.txt")

        results = {}
        threads = []  # Array of type: [workload.Workload]
        curr_workload_generator_index = 0
        for wl in self.workload_description:
            print(wl)
            rps = int(config["load_total"] * wl.rps_ratio)
            if wl.rps < 0:
                wl = wl._replace(rps=rps)
            if isinstance(wl.raw_payload, list):
                raw_payload = wl.raw_payload[config["iteration"] % len(wl.raw_payload)]
                wl = wl._replace(raw_payload=raw_payload)
            load_generators = []
            if len(self.machines) < 1:
                raise Exception("No machines for load generation, aborting")
            for _ in range(NUM_MACHINES_PER_WORKLOAD):
                load_generators.append(self.machines[curr_workload_generator_index])
                curr_workload_generator_index = (curr_workload_generator_index + 1) % len(self.machines)

            print(f"Generating workload for machines {load_generators}")
            load = workload.Workload(
                load_generators,
                self.target_nodes,
                wl,
                self.iter_outdir,
                f_stdout,
                f_stderr,
            )
            load.start()
            threads.append(load)

        all_destinations = []
        # dictionary of type: dict[str] = (workload.Workload, str)
        workload_command_summary_map = {}
        for num, thread in enumerate(threads):
            thread: workload.Workload = thread
            thread.join()
            commands = thread.get_commands()
            destinations = thread.fetch_results()
            assert len(commands) == len(destinations) == len(thread.uuids)
            for command, destination, uid in zip(commands, destinations, thread.uuids):
                workload_command_summary_map[str(uid)] = {
                    "command": command,
                    "workload_description": json.dumps(thread.workload, cls=BytesEncoder, indent=4),
                    "summary_file": destination,
                }

            print("Evaluating results from machines: {}".format(destinations))
            all_destinations += destinations
            results[num] = report.evaluate_summaries(destinations)

        with open(os.path.join(self.iter_outdir, "workload_command_summary_map.json"), "x") as map_file:
            map_file.write(json.dumps(workload_command_summary_map, cls=BytesEncoder, indent=4))

        aggregated = report.evaluate_summaries(all_destinations)
        return (results, aggregated)

    def run_iterations(self, iterations=None):
        """Exercise the experiment with specified iterations."""
        results = []
        rps_max = 0
        for i, d in enumerate(iterations):
            print(f"🚀 Running with total load {d}")
            config = {"load_total": d, "iteration": i}
            res, aggregated = self.run_experiment(config)

            duration = max([wl.start_delay + wl.duration for wl in self.workload_description])
            avg_succ_rate = aggregated.get_avg_success_rate(duration)
            latency = aggregated.percentiles[95] if aggregated.num_success > 0 else sys.float_info.max
            if (
                aggregated.failure_rate < workload_experiment.ALLOWABLE_FAILURE_RATE
                and latency < workload_experiment.ALLOWABLE_LATENCY
            ):
                if avg_succ_rate > rps_max:
                    rps_max = avg_succ_rate
            results.append((config, res, aggregated))
        data = [workloads for _, workloads, _ in results]
        num_workloads = len(self.workload_description)
        print(results)
        self.write_summary_file(
            "run_mixed_workload_experiment",
            {
                "is_update": FLAGS.use_updates,
                "rps_base": [rate for rate, _, _ in results],
                "failure_rate": [[d[i].failure_rate for d in data] for i in range(num_workloads)],
                "latency": [[d[i].percentiles[95] for d in data] for i in range(num_workloads)],
                "rps_max": rps_max,
                "labels": [
                    f"{d.get('canister', '')} - "
                    f"{d.get('rps_ratio', '')}% rps with "
                    f"{d.get('arguments', '')} @"
                    f"{d.get('start_delay', 0)}s for {d.get('duration', '')}s"
                    for d in self.raw_description["workload"]
                ],
                "description": self.raw_description["description"],
                "title": self.raw_description["title"],
            },
            iterations,
            "base requests / s",
            "mixed",
        )


if __name__ == "__main__":
    exp = MixedWorkloadExperiment()
    iterations = misc.get_iterations(FLAGS.target_rps, FLAGS.initial_rps, FLAGS.max_rps, FLAGS.increment_rps, 2)
    print(f"🚀 Running with iterations: {iterations}")
    exp.run_iterations(iterations)
    exp.end_experiment()
