
from dataclasses import dataclass, field

from core import Pos, Timestamp
from mechanics import CellContents, RawObservations


ObservationLog = dict[Timestamp, RawObservations]


@dataclass
class Logbook:
    latest_observations: RawObservations = field(default_factory=dict)
    observation_log: ObservationLog = field(default_factory=dict)
    last_observations_by_pos: dict[Pos, tuple[Timestamp, list[CellContents]]] = field(default_factory=dict)

    _latest_observations_at: Timestamp | None = None

    def add_latest_observations(self, now: Timestamp, observations: RawObservations) -> None:
        self.latest_observations = (
            observations if self._latest_observations_at is None or now > self._latest_observations_at else
            {**self.latest_observations, **observations} if now == self._latest_observations_at else
            self.latest_observations
        )
        self._latest_observations_at = now

        self.latest_observations = observations
        self.observation_log.setdefault(now, {}).update(observations)
        for pos, contents_list in observations.items():
            self.last_observations_by_pos[pos] = (now, contents_list)

    def copy_from(self, other: 'Logbook') -> None:
        if other._latest_observations_at is not None:
            self.add_latest_observations(other._latest_observations_at, other.latest_observations)

        for timestamp, raw_observations in other.observation_log.items():
            self.observation_log.setdefault(timestamp, {}).update(raw_observations)

        for pos, (timestamp, contents_list) in other.last_observations_by_pos.items():
            if not (
                pos in self.last_observations_by_pos
                and self.last_observations_by_pos[pos][0] >= timestamp
            ):
                self.last_observations_by_pos[pos] = (timestamp, contents_list)

    def clear(self) -> None:
        # TODO: maybe keep the very most recent observations
        self.observation_log.clear()
        self.last_observations_by_pos.clear()