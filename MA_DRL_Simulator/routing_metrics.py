import csv
import json
import os
from collections import Counter

import numpy as np


class RoutingMetrics:

    FILES = {
        "decisions": (
            "routing_decisions.csv",
            [
                "time_s", "block_id", "current_satellite",
                "next_hop", "decision_kind",
                "queue_length_before_enqueue"
            ],
        ),
        "arrivals": (
            "link_arrivals.csv",
            [
                "time_s", "block_id", "source",
                "destination", "link_type", "distance_km"
            ],
        ),
        "queues": (
            "queue_samples.csv",
            [
                "time_s", "num_queues", "total_queued",
                "mean_queue_length", "max_queue_length",
                "p95_queue_length", "nonempty_queues",
                "isl_queued", "gsl_up_queued", "gsl_down_queued"
            ],
        ),
        "anomalies": (
            "routing_anomalies.csv",
            ["time_s", "block_id", "reason", "definite_drop"],
        ),
    }

    def __init__(
        self, env, output_folder, light_speed,
        controller, custom_mode=False, queue_interval=0.01
    ):
        if queue_interval <= 0:
            raise ValueError("queue_interval must be positive")

        self.env = env
        self.folder = output_folder
        self.light_speed = float(light_speed)
        self.controller = str(controller)
        self.custom_mode = bool(custom_mode)
        self.queue_interval = float(queue_interval)
        self.files = {}
        self.writers = {}
        self.queue_hist = Counter()
        self.queue_observations = 0
        self.queue_sum = 0
        self.queue_max = 0
        self.queue_nonempty = 0
        self.queue_samples_count = 0
        self.decision_count = 0
        self.arrival_count = 0
        self.closed = False

        os.makedirs(self.folder, exist_ok=True)

        for key, (filename, fields) in self.FILES.items():
            handle = open(
                os.path.join(self.folder, filename),
                "w",
                newline="",
                encoding="utf-8",
                buffering=1024 * 1024,
            )
            self.files[key] = handle

            writer = csv.DictWriter(
                handle,
                fieldnames=fields,
                extrasaction="ignore"
            )
            writer.writeheader()
            self.writers[key] = writer

    def _write(self, key, **data):
        if self.closed:
            raise RuntimeError("RoutingMetrics already closed")
        self.writers[key].writerow(data)

    def close(self):
        if not self.closed:
            for handle in self.files.values():
                handle.flush()
                handle.close()
            self.closed = True

    def transmission_scheduled(
        self, block, source, destination,
        link_type, propagation_s
    ):

        if getattr(block, "eval_pending_link", None) is not None:
            self.anomaly(
                block,
                "OVERLAPPING_TRANSMISSION_RECORD",
                False
            )

        block.eval_pending_link = (
            str(source),
            str(destination),
            str(link_type),
            float(propagation_s) * self.light_speed,
        )

    def arrived(self, block, destination):


        pending = getattr(block, "eval_pending_link", None)

        if pending is None:
            self.anomaly(
                block,
                "ARRIVAL_WITHOUT_TRANSMISSION_RECORD",
                False
            )
            return

        source, expected_destination, link_type, distance_m = pending

        if str(destination) != expected_destination:
            self.anomaly(
                block,
                "ARRIVAL_DESTINATION_MISMATCH",
                False
            )

        block.eval_pending_link = None

        if link_type == "ISL":
            block.eval_isl_hops += 1
            block.eval_isl_distance_m += distance_m

            if str(destination) in block.eval_visited_satellites:
                block.eval_revisits += 1

            block.eval_visited_satellites.add(str(destination))

        elif link_type == "GSL":
            block.eval_gsl_hops += 1
            block.eval_gsl_distance_m += distance_m

            if str(destination) != str(block.destination.name):
                block.eval_visited_satellites.add(str(destination))

        block.eval_path.append(str(destination))
        self.arrival_count += 1

        self._write(
            "arrivals",
            time_s=float(self.env.now),
            block_id=block.ID,
            source=source,
            destination=destination,
            link_type=link_type,
            distance_km=distance_m / 1000.0,
        )

    def decision(self, block, satellite, next_hop):

        if next_hop == 0:
            target = block.destination.name
            kind = "DIRECT_GATEWAY"
            queue_len = len(satellite.sendBufferGT[1])

        elif isinstance(next_hop, (list, tuple)) and next_hop:
            target = next_hop[0]
            kind = "NEXT_SATELLITE"
            queue_len = None

            for buffer in (
                satellite.sendBufferSatsIntra
                + satellite.sendBufferSatsInter
            ):
                if str(buffer[2]) == str(target):
                    queue_len = len(buffer[1])
                    break

        else:
            target = str(next_hop)
            kind = "OTHER"
            queue_len = None

        self.decision_count += 1

        self._write(
            "decisions",
            time_s=float(self.env.now),
            block_id=block.ID,
            current_satellite=satellite.ID,
            next_hop=target,
            decision_kind=kind,
            queue_length_before_enqueue=queue_len,
        )

    def anomaly(self, block, reason, definite_drop=False):

        if definite_drop:
            block.eval_definite_drop = str(reason)

        self._write(
            "anomalies",
            time_s=float(self.env.now),
            block_id=block.ID,
            reason=reason,
            definite_drop=int(bool(definite_drop)),
        )

    def queue_monitor(self, earth):

        while True:
            lengths = []
            counts = {
                "ISL": 0,
                "GSL_UP": 0,
                "GSL_DOWN": 0
            }

            for gt in earth.gateways:
                n = len(gt.sendBuffer[1])
                lengths.append(n)
                counts["GSL_UP"] += n

            for plane in earth.LEO:
                for sat in plane.sats:

                    buffers = (
                        sat.sendBufferSatsIntra
                        + sat.sendBufferSatsInter
                    )

                    for buffer in buffers:
                        n = len(buffer[1])
                        lengths.append(n)
                        counts["ISL"] += n

                    n = len(sat.sendBufferGT[1])
                    lengths.append(n)
                    counts["GSL_DOWN"] += n

            total = sum(lengths)
            maximum = max(lengths, default=0)
            nonempty = sum(n > 0 for n in lengths)

            self.queue_hist.update(lengths)
            self.queue_observations += len(lengths)
            self.queue_sum += total
            self.queue_nonempty += nonempty
            self.queue_max = max(self.queue_max, maximum)
            self.queue_samples_count += 1

            self._write(
                "queues",
                time_s=float(self.env.now),
                num_queues=len(lengths),
                total_queued=total,
                mean_queue_length=(
                    float(np.mean(lengths)) if lengths else 0.0
                ),
                max_queue_length=maximum,
                p95_queue_length=(
                    float(np.percentile(lengths, 95))
                    if lengths else 0.0
                ),
                nonempty_queues=nonempty,
                isl_queued=counts["ISL"],
                gsl_up_queued=counts["GSL_UP"],
                gsl_down_queued=counts["GSL_DOWN"],
            )

            yield self.env.timeout(self.queue_interval)

    def _hist_percentile(self, fraction):
        if not self.queue_observations:
            return None

        threshold = (self.queue_observations - 1) * fraction
        seen = 0

        for value, count in sorted(self.queue_hist.items()):
            seen += count
            if seen - 1 >= threshold:
                return value

        return max(self.queue_hist)

    @staticmethod
    def _mean(values):
        return float(np.mean(values)) if values else None

    def export(self, created, received, lost_counter, end_s, config):


        delivered_ids = {id(block) for block in received}
        rows = []
        status_counts = Counter()

        for block in created:

            actually_delivered = id(block) in delivered_ids
            terminal_status = getattr(
                block, "custom_terminal_status", None
            )
            terminal_reason = getattr(
                block, "custom_terminal_reason", None
            )
            definite_drop = getattr(
                block, "eval_definite_drop", None
            )

            if self.custom_mode:

                if terminal_status == "DELIVERED" and not actually_delivered:
                    raise RuntimeError(
                        f"Block {block.ID}: DELIVERED but not received"
                    )

                if actually_delivered and terminal_status != "DELIVERED":
                    raise RuntimeError(
                        f"Block {block.ID}: received but status "
                        f"is {terminal_status}"
                    )

                if actually_delivered:
                    status = "DELIVERED"
                elif terminal_status is None:
                    status = "PENDING"
                else:
                    status = "FAILED"

                if status == "FAILED":
                    failure_reason = (
                        f"{terminal_status}:{terminal_reason}"
                        if terminal_reason else terminal_status or ""
                    )
                else:
                    failure_reason = ""

            else:
                if actually_delivered:
                    status = "DELIVERED"
                elif definite_drop:
                    status = "FAILED"
                else:
                    status = "PENDING"

                failure_reason = (
                    str(definite_drop)
                    if status == "FAILED" else ""
                )

            status_counts[terminal_status or "UNRESOLVED"] += 1

            first_tx = block.timeAtFirstTransmission
            filled = block.timeAtFull

            uplink_queue = (
                max(0.0, float(first_tx - filled))
                if first_tx is not None and filled is not None
                else 0.0
            )

            satellite_queue = sum(
                max(0.0, float(sent - arrived))
                for arrived, sent in zip(
                    block.checkPoints,
                    block.checkPointsSend
                )
            )

            if status == "DELIVERED":
                end_at = getattr(
                    block, "eval_delivered_at", None
                )
            elif self.custom_mode and terminal_status:
                end_at = getattr(
                    block, "custom_terminal_time", None
                )
            else:
                end_at = float(end_s)

            rows.append({
                "block_id": block.ID,
                "source": block.source.name,
                "destination": block.destination.name,
                "status": status,
                "failure_reason": failure_reason,
                "creation_s": float(block.creationTime),
                "filled_s": filled,
                "first_transmission_s": first_tx,
                "end_s": end_at,
                "isl_hops": block.eval_isl_hops,
                "gsl_hops": block.eval_gsl_hops,
                "isl_distance_km": (
                    block.eval_isl_distance_m / 1000.0
                ),
                "gsl_distance_km": (
                    block.eval_gsl_distance_m / 1000.0
                ),
                "total_distance_km": (
                    block.eval_isl_distance_m
                    + block.eval_gsl_distance_m
                ) / 1000.0,
                "queue_wait_observed_s": (
                    uplink_queue + satellite_queue
                ),
                "revisits": block.eval_revisits,
                "path": " -> ".join(block.eval_path),
            })

        filename = os.path.join(
            self.folder, "packet_results.csv"
        )

        if rows:
            with open(
                filename, "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(rows[0])
                )
                writer.writeheader()
                writer.writerows(rows)
        else:
            with open(
                filename, "w", encoding="utf-8"
            ) as handle:
                handle.write("block_id,status\n")

        delivered_rows = [
            row for row in rows
            if row["status"] == "DELIVERED"
        ]

        delivered = len(delivered_rows)

        failed = sum(
            row["status"] == "FAILED" for row in rows
        )

        pending = sum(
            row["status"] == "PENDING" for row in rows
        )

        resolved = delivered + failed

        summary = {
            "controller": self.controller,
            "native_simulation": not self.custom_mode,
            "custom_mode": self.custom_mode,
            "simulation_end_s": float(end_s),
            "created_filled": len(rows),
            "delivered": delivered,
            "confirmed_failed": failed,
            "pending_or_unclassified": pending,

            "success_rate_resolved_pct": (
                100.0 * delivered / resolved
                if resolved else None
            ),

            "delivery_rate_all_pct": (
                100.0 * delivered / len(rows)
                if rows else None
            ),

            "original_lost_blocks_counter": int(lost_counter),

            "note": (
                "CUSTOM: FAILED includes routing failures "
                "and deadline timeouts. A timeout is not "
                "proof of a physical drop."
                if self.custom_mode else
                "PENDING includes packets not delivered "
                "and not definitively marked as dropped. "
                "Resolved success may overestimate performance."
            ),

            "mean_delivered_distance_km": self._mean([
                r["total_distance_km"] for r in delivered_rows
            ]),

            "p95_delivered_distance_km": (
                float(np.percentile(
                    [r["total_distance_km"]
                     for r in delivered_rows], 95
                ))
                if delivered_rows else None
            ),

            "mean_delivered_isl_hops": self._mean([
                r["isl_hops"] for r in delivered_rows
            ]),

            "mean_delivered_queue_wait_s": self._mean([
                r["queue_wait_observed_s"]
                for r in delivered_rows
            ]),

            "mean_delivered_latency_from_full_s": self._mean([
                r["end_s"] - r["filled_s"]
                for r in delivered_rows
                if r["end_s"] is not None
                and r["filled_s"] is not None
            ]),

            "queue_sample_interval_s": self.queue_interval,
            "queue_samples": self.queue_samples_count,

            "queue_mean_over_all_links_and_times": (
                self.queue_sum / self.queue_observations
                if self.queue_observations else None
            ),

            "queue_p95_over_all_links_and_times":
                self._hist_percentile(0.95),

            "queue_max": self.queue_max,

            "fraction_queue_observations_nonempty": (
                self.queue_nonempty / self.queue_observations
                if self.queue_observations else None
            ),

            "routing_decisions": self.decision_count,
            "completed_link_arrivals": self.arrival_count,
            "configuration": config,
        }

        if self.custom_mode:
            summary["custom_terminal_status_counts"] = dict(
                status_counts
            )

            summary["custom_timeout_count"] = sum(
                status_counts[key]
                for key in ("PACKET_TIMEOUT", "GLOBAL_TIMEOUT")
            )

            summary["custom_routing_failed_count"] = (
                status_counts["ROUTING_FAILED"]
            )

        try:
            with open(
                os.path.join(
                    self.folder, "simulation_summary.json"
                ),
                "w",
                encoding="utf-8"
            ) as handle:
                json.dump(
                    summary,
                    handle,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False
                )
        finally:
            self.close()

        return summary


def identify_controller(
    use_rigoletto, pathing, online_phase=False
):

    if use_rigoletto:
        return "RIGOLETTO"

    if pathing == "Deep Q-Learning":
        return (
            "DECENTRALIZED_DDQN"
            if online_phase else "CENTRALIZED_DDQN"
        )

    return (
        str(pathing)
        .upper()
        .replace(" ", "_")
        .replace("-", "_")
    )


def start_routing_metrics(earth, env, output_path, gt_number, light_speed, controller, custom_mode, queue_interval=0.01):

    mode = "custom" if custom_mode else "native"

    folder = os.path.join(
        output_path,
        "routing_metrics",
        controller,
        mode,
        f"{gt_number}GTs"
    )

    metrics = RoutingMetrics(
        env,
        folder,
        light_speed,
        controller,
        custom_mode=custom_mode,
        queue_interval=queue_interval
    )

    earth.eval_metrics = metrics
    env.process(metrics.queue_monitor(earth))
    return metrics


def finalize_routing_metrics(
    earth, created, received, end_s, config
):
    metrics = getattr(
        earth, "eval_metrics", None
    )

    if metrics is None:
        return None

    summary = metrics.export(
        created,
        received,
        earth.lostBlocks,
        end_s,
        config=config
    )

    print(
        f"\n=== ROUTING METRICS | "
        f"{summary['controller']} | "
        f"{'CUSTOM' if summary['custom_mode'] else 'NATIVE'} ==="
    )

    print("Generati:", summary["created_filled"])
    print("Consegnati:", summary["delivered"])
    print("Falliti/timeout:", summary["confirmed_failed"])
    print("Pendenti:", summary["pending_or_unclassified"])
    print("Delivery rate (%):", summary["delivery_rate_all_pct"])
    print("Success rate resolved (%):", summary["success_rate_resolved_pct"])
    print("Distanza media (km):", summary["mean_delivered_distance_km"])
    print("Hop ISL medi:", summary["mean_delivered_isl_hops"])
    print("Attesa media in coda (s):", summary["mean_delivered_queue_wait_s"])
    print("Latenza media (s):", summary["mean_delivered_latency_from_full_s"])
    print("Lunghezza media code:", summary["queue_mean_over_all_links_and_times"])
    print("Picco congestione:", summary["queue_max"])
    print(
        "File:",
        os.path.join(
            metrics.folder, "simulation_summary.json"
        )
    )

    return summary
