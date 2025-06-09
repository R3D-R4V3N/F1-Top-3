import pandas as pd
import pytest

from prepare_data import compute_overtakes, compute_track_overtake_index


def test_compute_overtakes_basic(monkeypatch):
    """Basic sanity check for overtake counting with pit stop removal."""

    laps = pd.DataFrame({
        'raceId': [1, 1, 1, 1, 1, 1, 1],
        'driverId': ['drv'] * 7,
        'lap': [1, 2, 3, 4, 5, 6, 7],
        'position': [5, 4, 4, 3, 3, 5, 4],
        'sc_flag': [0, 0, 1, 0, 0, 0, 0],
        'track_id': ['A'] * 7,
    })
    pits = pd.DataFrame({'driverId': ['drv'], 'lap': [5]})

    # use a fixed track index to avoid reading files
    monkeypatch.setattr(
        'prepare_data.compute_track_overtake_index',
        lambda laps_folder, pit_folder, sc_events=None: pd.Series({'A': 0.5})
    )

    result = compute_overtakes(laps, pits)

    assert result.loc[0, 'overtakes_count'] == 2
    assert result.loc[0, 'overtakes_rate'] == pytest.approx(2 / 4)
    assert result.loc[0, 'overtakes_post_SC'] == 1
    assert result.loc[0, 'overtakes_last10'] == 2
    assert result.loc[0, 'track_overtake_index'] == 0.5


def test_track_overtake_index_and_mapping(tmp_path, monkeypatch):
    """Full integration: compute track index from csv files and map in compute_overtakes."""

    data_dir = tmp_path / 'data'
    data_dir.mkdir()

    # Helper to write lap/pit files
    def write_race(name, race_id, driver, positions, track):
        df_lap = pd.DataFrame({
            'raceId': [race_id] * len(positions),
            'driverId': [driver] * len(positions),
            'lap': list(range(1, len(positions) + 1)),
            'position': positions,
            'track_id': [track] * len(positions),
        })
        df_lap.to_csv(data_dir / f'{name}_laps.csv', index=False)
        # empty pit stop file
        pd.DataFrame({'driverId': [], 'lap': []}).to_csv(data_dir / f'{name}_pitstops.csv', index=False)

    # Track A races with no overtakes
    write_race('A1', 1, 'd1', [5, 5, 5], 'A')
    write_race('A2', 2, 'd2', [4, 4, 4], 'A')
    # Track B races with constant overtakes
    write_race('B1', 3, 'd3', [5, 4, 3], 'B')
    write_race('B2', 4, 'd4', [4, 3, 2], 'B')

    track_index = compute_track_overtake_index(data_dir, data_dir)

    assert track_index['A'] == pytest.approx(0.0)
    assert track_index['B'] == pytest.approx(2 / 3)

    # use one race to check mapping inside compute_overtakes
    laps_b1 = pd.read_csv(data_dir / 'B1_laps.csv')
    pits_b1 = pd.read_csv(data_dir / 'B1_pitstops.csv')

    monkeypatch.setattr(
        'prepare_data.compute_track_overtake_index',
        lambda laps_folder, pit_folder, sc_events=None: track_index
    )

    res = compute_overtakes(laps_b1, pits_b1)
    assert 'track_overtake_index' in res.columns
    assert res.loc[0, 'track_overtake_index'] == pytest.approx(track_index['B'])
