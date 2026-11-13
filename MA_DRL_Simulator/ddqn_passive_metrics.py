import csv
import json
import os

from collections import Counter

import numpy as np

"""
file per la gestione del salvataggio in csv delle metriche di congestione dell'algoritmo originale usato nel simulatore
"""

class PassiveDDQNMetrics:

    FILES = {
        "arrivals": (
            "link_arrivals.csv",
            [
                "time_s",
                "block_id",
                "source",
                "destination",
                "link_type",
                "distance_km"
            ]
        ),

        "queues": (
            "queue_samples.csv",
            [
                "time_s",
                "num_queues",
                "total_queued",
                "mean_queue_length",
                "max_queue_length",
                "p95_queue_length",
                "nonempty_queues",
                "isl_queued",
                "gsl_up_queued",
                "gsl_down_queued"
            ]
        ),

        "anomalies": (
            "routing_anomalies.csv",
            [
                "time_s",
                "block_id",
                "reason",
                "definite_drop"
            ]
        )
    }

    def __init__(
        self,
        env,
        output_folder,
        light_speed,
        queue_interval=0.5
    ):

        if queue_interval <= 0:
            raise ValueError(
                "queue_interval must be positive"
            )

        self.env = env
        self.folder = output_folder
        self.light_speed = light_speed
        self.queue_interval = queue_interval

        self.files = {}
        self.writers = {}

        # Statistiche congestione
        self.queue_hist = Counter()
        self.queue_observations = 0
        self.queue_sum = 0
        self.queue_max = 0
        self.queue_nonempty = 0
        self.queue_samples_count = 0

        # Numero di collegamenti completati
        self.arrival_count = 0

        os.makedirs(
            self.folder,
            exist_ok=True
        )

        for key, (filename, fields) in self.FILES.items():

            file = open(
                os.path.join(
                    self.folder,
                    filename
                ),
                "w",
                newline="",
                encoding="utf-8",
                buffering=1024 * 1024
            )

            self.files[key] = file

            writer = csv.DictWriter(
                file,
                fieldnames=fields,
                extrasaction="ignore"
            )

            writer.writeheader()

            self.writers[key] = writer

    def _write(self, key, **data):

        self.writers[key].writerow(data)

    def transmission_scheduled(
        self,
        block,
        source,
        destination,
        link_type,
        propagation_s
    ):

        # Il collegamento viene registrato come completato soltanto all'arrivo del DataBlock.

        distance_m = (
            float(propagation_s)
            * self.light_speed
        )

        block.eval_pending_link = (
            str(source),
            str(destination),
            str(link_type),
            distance_m
        )

    def arrived(self, block, destination):

        pending = getattr(
            block,
            "eval_pending_link",
            None
        )

        if pending is None:

            self.anomaly(
                block,
                "ARRIVAL_WITHOUT_TRANSMISSION_RECORD",
                False
            )

            return

        (
            source,
            expected_destination,
            link_type,
            distance_m
        ) = pending

        if str(destination) != expected_destination:

            self.anomaly(
                block,
                "ARRIVAL_DESTINATION_MISMATCH",
                False
            )

        block.eval_pending_link = None

        # Collegamento tra satelliti
        if link_type == "ISL":

            block.eval_isl_hops += 1

            block.eval_isl_distance_m += distance_m

            if (
                str(destination)
                in block.eval_visited_satellites
            ):

                block.eval_revisits += 1

            block.eval_visited_satellites.add(
                str(destination)
            )

        # Collegamento satellite-ground station
        elif link_type == "GSL":

            block.eval_gsl_hops += 1

            block.eval_gsl_distance_m += distance_m

            if (
                str(destination)
                != str(block.destination.name)
            ):

                block.eval_visited_satellites.add(
                    str(destination)
                )

        block.eval_path.append(
            str(destination)
        )

        self.arrival_count += 1

        self._write(
            "arrivals",
            time_s=float(self.env.now),
            block_id=block.ID,
            source=source,
            destination=destination,
            link_type=link_type,
            distance_km=distance_m / 1000.0
        )

    def anomaly(
        self,
        block,
        reason,
        definite_drop=False
    ):

        if definite_drop:

            block.eval_definite_drop = str(reason)

        self._write(
            "anomalies",
            time_s=float(self.env.now),
            block_id=block.ID,
            reason=reason,
            definite_drop=int(bool(definite_drop))
        )

    def queue_monitor(self, earth):

        while True:

            lengths = []

            counts = {
                "ISL": 0,
                "GSL_UP": 0,
                "GSL_DOWN": 0
            }

            # Ground stations
            for gt in earth.gateways:

                n = len(gt.sendBuffer[1])

                lengths.append(n)

                counts["GSL_UP"] += n

            # Satelliti
            for plane in earth.LEO:

                for sat in plane.sats:

                    # Collegamenti ISL
                    for buffer in (
                        sat.sendBufferSatsIntra
                        + sat.sendBufferSatsInter
                    ):

                        n = len(buffer[1])

                        lengths.append(n)

                        counts["ISL"] += n

                    # Collegamenti GSL verso le ground stations
                    n = len(sat.sendBufferGT[1])

                    lengths.append(n)

                    counts["GSL_DOWN"] += n

            # Statistiche aggregate
            self.queue_hist.update(lengths)

            self.queue_observations += len(lengths)

            self.queue_sum += sum(lengths)

            self.queue_nonempty += sum(
                n > 0
                for n in lengths
            )

            self.queue_max = max(
                self.queue_max,
                max(lengths, default=0)
            )

            self.queue_samples_count += 1

            # Salvataggio campione
            self._write(
                "queues",
                time_s=float(self.env.now),
                num_queues=len(lengths),
                total_queued=sum(lengths),
                mean_queue_length=(
                    float(np.mean(lengths))
                    if lengths else 0.0
                ),
                max_queue_length=max(
                    lengths,
                    default=0
                ),
                p95_queue_length=(
                    float(np.percentile(lengths, 95))
                    if lengths else 0.0
                ),
                nonempty_queues=sum(
                    n > 0
                    for n in lengths
                ),
                isl_queued=counts["ISL"],
                gsl_up_queued=counts["GSL_UP"],
                gsl_down_queued=counts["GSL_DOWN"]
            )

            yield self.env.timeout(
                self.queue_interval
            )

    # osservazioni sullo stato delle code
    def _hist_percentile(self, fraction):

        if not self.queue_observations:
            return None

        threshold = (
            self.queue_observations - 1
        ) * fraction

        seen = 0

        for value, count in sorted(
            self.queue_hist.items()
        ):

            seen += count

            if seen - 1 >= threshold:
                return value

        return max(self.queue_hist)

    @staticmethod
    def _mean(values):

        return (
            float(np.mean(values))
            if values
            else None
        )

    # salvataggio dei risultati

    def export(
        self,
        created,
        received,
        lost_counter,
        end_s,
        config
    ):

        delivered_ids = {
            id(block)
            for block in received
        }

        rows = []

        for block in created:

            delivered = (
                id(block) in delivered_ids
            )

            definite_drop = getattr(
                block,
                "eval_definite_drop",
                None
            )

            if delivered:
                status = "DELIVERED"

            elif definite_drop:
                status = "FAILED"

            else:
                status = "PENDING"

            first_tx = block.timeAtFirstTransmission
            filled = block.timeAtFull

            # Attesa prima della prima trasmissione
            uplink_queue = (
                max(
                    0.0,
                    float(first_tx - filled)
                )
                if (
                    first_tx is not None
                    and filled is not None
                )
                else 0.0
            )

            # Attesa osservata nei satelliti
            satellite_queue = sum(
                max(
                    0.0,
                    float(sent - arrived)
                )
                for arrived, sent in zip(
                    block.checkPoints,
                    block.checkPointsSend
                )
            )

            total_distance_km = (
                block.eval_isl_distance_m
                + block.eval_gsl_distance_m
            ) / 1000.0

            rows.append({
                "block_id": block.ID,
                "source": block.source.name,
                "destination": block.destination.name,
                "status": status,
                "failure_reason": (
                    definite_drop
                    if status == "FAILED"
                    else ""
                ),
                "creation_s": float(
                    block.creationTime
                ),
                "filled_s": filled,
                "first_transmission_s": first_tx,
                "end_s": (
                    getattr(
                        block,
                        "eval_delivered_at",
                        None
                    )
                    if delivered
                    else float(end_s)
                ),
                "isl_hops": block.eval_isl_hops,
                "gsl_hops": block.eval_gsl_hops,
                "isl_distance_km": (
                    block.eval_isl_distance_m / 1000.0
                ),
                "gsl_distance_km": (
                    block.eval_gsl_distance_m / 1000.0
                ),
                "total_distance_km": total_distance_km,
                "queue_wait_observed_s": (
                    uplink_queue + satellite_queue
                ),
                "revisits": block.eval_revisits,
                "path": " -> ".join(
                    block.eval_path
                )
            })

        filename = os.path.join(
            self.folder,
            "packet_results.csv"
        )

        if rows:

            with open(
                filename,
                "w",
                newline="",
                encoding="utf-8"
            ) as file:

                writer = csv.DictWriter(
                    file,
                    fieldnames=list(rows[0].keys())
                )

                writer.writeheader()

                writer.writerows(rows)

        else:

            with open(
                filename,
                "w",
                encoding="utf-8"
            ) as file:

                file.write(
                    "block_id,status\n"
                )

        # Statistiche sulla consegna
        delivered_rows = [
            row
            for row in rows
            if row["status"] == "DELIVERED"
        ]

        delivered = len(delivered_rows)

        failed = sum(
            row["status"] == "FAILED"
            for row in rows
        )

        pending = sum(
            row["status"] == "PENDING"
            for row in rows
        )

        resolved = delivered + failed

        # Esclude pacchetti non arrivati per timeout della simulazione
        success_rate = (
            100.0 * delivered / resolved
            if resolved > 0
            else None
        )

        # % consegna
        delivery_rate = (
            100.0 * delivered / len(rows)
            if rows
            else None
        )

        # riepilogo
        summary = {
            "controller": "CENTRALIZED_DDQN",
            "native_simulation": True,
            "custom_mode": False,

            "simulation_end_s": float(end_s),

            "created_filled": len(rows),
            "delivered": delivered,
            "confirmed_failed": failed,
            "pending_or_unclassified": pending,

            "success_rate_resolved_pct": success_rate,
            "delivery_rate_all_pct": delivery_rate,

            "original_lost_blocks_counter": int(
                lost_counter
            ),

            "note": (
                "PENDING includes packets not delivered "
                "and not definitively marked as dropped. "
                "These cases may include unfinished "
                "transmissions and unclassified losses. "
                "The resolved success rate may therefore "
                "overestimate real performance."
            ),

            "mean_delivered_distance_km": self._mean(
                [
                    row["total_distance_km"]
                    for row in delivered_rows
                ]
            ),

            "p95_delivered_distance_km": (
                float(
                    np.percentile(
                        [
                            row["total_distance_km"]
                            for row in delivered_rows
                        ],
                        95
                    )
                )
                if delivered_rows
                else None
            ),

            "mean_delivered_isl_hops": self._mean(
                [
                    row["isl_hops"]
                    for row in delivered_rows
                ]
            ),

            "mean_delivered_queue_wait_s": self._mean(
                [
                    row["queue_wait_observed_s"]
                    for row in delivered_rows
                ]
            ),

            "mean_delivered_latency_from_full_s": self._mean(
                [
                    row["end_s"] - row["filled_s"]
                    for row in delivered_rows
                    if (
                        row["end_s"] is not None
                        and row["filled_s"] is not None
                    )
                ]
            ),

            "queue_sample_interval_s": self.queue_interval,
            "queue_samples": self.queue_samples_count,

            "queue_mean_over_all_links_and_times": (
                self.queue_sum / self.queue_observations
                if self.queue_observations
                else None
            ),

            "queue_p95_over_all_links_and_times": (
                self._hist_percentile(0.95)
            ),

            "queue_max": self.queue_max,

            "fraction_queue_observations_nonempty": (
                self.queue_nonempty
                / self.queue_observations
                if self.queue_observations
                else None
            ),

            "completed_link_arrivals": self.arrival_count,

            "configuration": config
        }

        
        with open(
            os.path.join(
                self.folder,
                "simulation_summary.json"
            ),
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                summary,
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False
            )

        for file in self.files.values():

            file.flush()
            file.close()

        return summary