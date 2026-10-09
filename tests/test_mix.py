from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile

from converse_diarization.mix import (
    MIX_INDEX,
    MIX_POLICY,
    MIX_SAMPLE_RATE,
    MixError,
    ensure_mix,
    load_index,
    mix_arrays,
    mix_dataset,
    mix_dir,
)
from tests.fixtures import make_dataset


def test_mix_arrays_delays_by_offset_and_pads():
    a = np.ones(4, dtype=np.float32) * 0.25
    b = np.ones(2, dtype=np.float32) * 0.25
    mixed = mix_arrays([(a, 0.0), (b, 1.0)], sample_rate=2)  # b starts two samples in
    assert mixed.tolist() == pytest.approx([0.25, 0.25, 0.5, 0.5])


def test_mix_arrays_only_scales_when_clipping():
    quiet = mix_arrays([(np.full(3, 0.4, dtype=np.float32), 0.0), (np.full(3, 0.4, dtype=np.float32), 0.0)])
    assert quiet.max() == pytest.approx(0.8)
    loud = mix_arrays([(np.full(3, 0.9, dtype=np.float32), 0.0), (np.full(3, 0.9, dtype=np.float32), 0.0)])
    assert loud.max() == pytest.approx(1.0)
    with pytest.raises(MixError):
        mix_arrays([(np.zeros(3, dtype=np.float32), -0.1)])
    with pytest.raises(MixError):
        mix_arrays([])


def test_mix_dataset_builds_aligned_16k_wavs_and_index(tmp_path):
    dataset = make_dataset(tmp_path)
    results = mix_dataset(dataset)
    assert [r.conversation for r in results] == [c.id for c in dataset.conversations]
    s1 = next(r for r in results if r.conversation.endswith("s1"))
    samples, rate = soundfile.read(s1.path, dtype="float32")
    assert rate == MIX_SAMPLE_RATE
    # second track starts 0.5 s late, so the mix is 4.5 s long
    assert s1.duration_s == pytest.approx(4.5, abs=0.01)
    assert len(samples) == s1.sample_count
    # the first half second only carries the first track's tone (amplitude 0.5 -> no scaling)
    assert 0.45 < np.abs(samples[: MIX_SAMPLE_RATE // 2]).max() <= 0.51
    # after 0.5 s both tones are present and the sum peaks above a single track
    assert np.abs(samples[MIX_SAMPLE_RATE // 2 : MIX_SAMPLE_RATE]).max() > 0.6
    index = json.loads((mix_dir(dataset) / MIX_INDEX).read_text(encoding="utf-8"))
    assert index["policy"]["id"] == MIX_POLICY and len(index["mixes"]) == 2
    assert index["dataset_digest"] == dataset.digest()
    assert load_index(dataset)[s1.conversation]["sha256"] == s1.sha256


def test_ensure_mix_reuses_a_verified_mix_and_rebuilds_a_damaged_one(tmp_path):
    dataset = make_dataset(tmp_path)
    conversation = dataset.conversations[0]
    first = ensure_mix(dataset, conversation)
    mtime = first.path.stat().st_mtime_ns
    second = ensure_mix(dataset, conversation)
    assert second.sha256 == first.sha256 and second.path.stat().st_mtime_ns == mtime
    first.path.write_bytes(b"damaged")
    third = ensure_mix(dataset, conversation)
    assert third.sha256 == first.sha256 and first.path.stat().st_size > 100


def test_mix_refuses_audio_that_does_not_match_the_manifest(tmp_path):
    dataset = make_dataset(tmp_path)
    conversation = dataset.conversations[0]
    conversation.tracks[0].audio_path(dataset.root).write_bytes(b"not flac")
    with pytest.raises(MixError, match="does not match the manifest"):
        ensure_mix(dataset, conversation)
