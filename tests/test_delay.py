import unittest

import numpy as np

from ball_catch.config import EnvConfig
from ball_catch.delay import (
    BurstDelay,
    FixedDelay,
    Frame,
    FrameQueue,
    JitterDelay,
    SequenceDelay,
    make_delay_model,
)


def frame(index, capture_time, delay):
    return Frame(
        index=index,
        capture_time=capture_time,
        arrival_time=capture_time + delay,
        features=np.zeros(3),
        valid=True,
    )


class DelayModelTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_fixed_delay_is_constant(self):
        model = FixedDelay(0.1)
        model.reset(self.rng, flight_time=1.0)
        self.assertEqual([model.sample(i, i / 30, self.rng) for i in range(5)], [0.1] * 5)

    def test_jitter_stays_within_bounds(self):
        model = JitterDelay(mean=0.08, std=0.04, d_max=0.2)
        samples = np.array([model.sample(i, i / 30, self.rng) for i in range(3000)])
        self.assertGreaterEqual(samples.min(), 0.0)
        self.assertLessEqual(samples.max(), 0.2)

    def test_jitter_is_centred_on_its_mean_and_varies(self):
        model = JitterDelay(mean=0.08, std=0.04, d_max=0.2)
        samples = np.array([model.sample(i, i / 30, self.rng) for i in range(3000)])
        self.assertAlmostEqual(samples.mean(), 0.08, delta=0.01)
        self.assertGreater(samples.std(), 0.02)

    def test_jitter_truncation_is_not_clipping(self):
        # A wide Gaussian must be resampled into range, not piled up on the bounds.
        model = JitterDelay(mean=0.1, std=1.0, d_max=0.2)
        samples = np.array([model.sample(i, 0.0, self.rng) for i in range(2000)])
        self.assertLess(np.mean(samples == 0.0) + np.mean(samples == 0.2), 0.01)

    def test_burst_stalls_consecutive_frames_from_the_requested_flight_fraction(self):
        model = BurstDelay(FixedDelay(0.02), start_fraction=0.5, n_frames=3, stall_delay=0.2)
        model.reset(self.rng, flight_time=1.0)
        delays = [model.sample(i, i / 30, self.rng) for i in range(30)]
        stalled = [i for i, d in enumerate(delays) if d == 0.2]
        self.assertEqual(stalled, [15, 16, 17])
        self.assertTrue(all(d == 0.02 for i, d in enumerate(delays) if i not in stalled))

    def test_burst_position_scales_with_flight_time_after_reset(self):
        model = BurstDelay(FixedDelay(0.02), start_fraction=0.5, n_frames=2, stall_delay=0.2)
        model.reset(self.rng, flight_time=1.0)
        [model.sample(i, i / 30, self.rng) for i in range(30)]
        model.reset(self.rng, flight_time=0.6)
        delays = [model.sample(i, i / 30, self.rng) for i in range(30)]
        self.assertEqual([i for i, d in enumerate(delays) if d == 0.2], [9, 10])

    def test_burst_with_random_start_is_reproducible(self):
        def stalled_frames(seed):
            rng = np.random.default_rng(seed)
            model = BurstDelay(FixedDelay(0.0), start_fraction=None, n_frames=3, stall_delay=0.2)
            model.reset(rng, flight_time=1.0)
            return [i for i in range(30) if model.sample(i, i / 30, rng) == 0.2]

        self.assertEqual(stalled_frames(5), stalled_frames(5))
        self.assertEqual(len(stalled_frames(5)), 3)

    def test_sequence_replays_and_cycles(self):
        model = SequenceDelay([0.1, 0.0, 0.05])
        model.reset(self.rng, flight_time=1.0)
        self.assertEqual(
            [model.sample(i, i / 30, self.rng) for i in range(5)], [0.1, 0.0, 0.05, 0.1, 0.0]
        )

    def test_make_delay_model_builds_named_presets(self):
        cfg = EnvConfig()
        self.assertEqual(make_delay_model("none", cfg).sample(0, 0.0, self.rng), 0.0)
        self.assertEqual(make_delay_model("fixed", cfg).sample(0, 0.0, self.rng), cfg.jitter_mean)
        self.assertEqual(make_delay_model(0.15, cfg).sample(0, 0.0, self.rng), 0.15)
        self.assertIsInstance(make_delay_model("jitter", cfg), JitterDelay)
        self.assertIsInstance(make_delay_model("burst", cfg), BurstDelay)

    def test_make_delay_model_passes_instances_through(self):
        model = FixedDelay(0.03)
        self.assertIs(make_delay_model(model, EnvConfig()), model)

    def test_make_delay_model_rejects_unknown_names(self):
        with self.assertRaises(ValueError):
            make_delay_model("teleport", EnvConfig())


class FrameQueueTest(unittest.TestCase):
    def test_frame_is_not_delivered_before_it_arrives(self):
        queue = FrameQueue()
        queue.push(frame(0, capture_time=0.0, delay=0.1))
        self.assertIsNone(queue.deliver(0.05))
        self.assertIsNone(queue.latest)

    def test_frame_is_delivered_exactly_once(self):
        queue = FrameQueue()
        queue.push(frame(0, capture_time=0.0, delay=0.1))
        delivered = queue.deliver(0.12)
        self.assertEqual(delivered.index, 0)
        self.assertEqual(delivered.status, "accepted")
        self.assertIs(queue.latest, delivered)
        self.assertIsNone(queue.deliver(0.14))
        self.assertIs(queue.latest, delivered)

    def test_frame_arriving_on_a_control_tick_is_delivered_at_that_tick(self):
        queue = FrameQueue()
        queue.push(frame(0, capture_time=0.0, delay=0.1))
        self.assertIsNotNone(queue.deliver(6 * (1.0 / 60.0)))  # 0.1 s up to rounding

    def test_older_frame_arriving_late_is_discarded(self):
        queue = FrameQueue()
        slow = frame(0, capture_time=0.0, delay=0.2)
        fast = frame(1, capture_time=1 / 30, delay=0.0)
        queue.push(slow)
        queue.push(fast)
        self.assertEqual(queue.deliver(0.05).index, 1)
        self.assertIsNone(queue.deliver(0.25))
        self.assertEqual(slow.status, "discarded")
        self.assertEqual(queue.latest.index, 1)
        self.assertEqual(queue.discarded_count, 1)
        self.assertEqual(queue.accepted_count, 1)

    def test_only_the_newest_of_several_simultaneous_arrivals_is_accepted(self):
        queue = FrameQueue()
        frames = [frame(i, capture_time=i / 30, delay=0.1 - i / 30) for i in range(3)]
        for f in frames:
            queue.push(f)
        self.assertEqual(queue.deliver(0.11).index, 2)
        self.assertEqual([f.status for f in frames], ["discarded", "discarded", "accepted"])

    def test_capture_interval_is_zero_until_two_frames_were_accepted(self):
        queue = FrameQueue()
        self.assertEqual(queue.capture_interval, 0.0)
        queue.push(frame(0, capture_time=0.0, delay=0.0))
        queue.deliver(0.0)
        self.assertEqual(queue.capture_interval, 0.0)

    def test_capture_interval_spans_discarded_frames(self):
        queue = FrameQueue()
        queue.push(frame(0, capture_time=0.0, delay=0.0))
        queue.push(frame(1, capture_time=1 / 30, delay=0.5))  # arrives far too late
        queue.push(frame(2, capture_time=2 / 30, delay=0.0))
        queue.deliver(0.0)
        queue.deliver(2 / 30)
        self.assertAlmostEqual(queue.capture_interval, 2 / 30)

    def test_reset_empties_the_queue(self):
        queue = FrameQueue()
        queue.push(frame(0, capture_time=0.0, delay=0.0))
        queue.deliver(0.0)
        queue.reset()
        self.assertIsNone(queue.latest)
        self.assertEqual(queue.frames, [])
        self.assertEqual(queue.accepted_count, 0)
        self.assertIsNone(queue.deliver(1.0))


if __name__ == "__main__":
    unittest.main()
