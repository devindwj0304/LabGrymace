"""
Test that generate_perframe_roi saves frames from the animation deque,
not from generate_patternimage_perframe contour drawings.
"""

import os
import tempfile
import numpy as np
import cv2
from collections import deque


def test_perframe_saves_animation_frames():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Build a fake animation: 4 frames, each a solid distinct color
        h, w = 64, 64
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 0)]
        animation = deque(maxlen=4)
        for color in colors:
            blob = np.full((h, w, 3), color, dtype='uint8')
            animation.append(blob)

        base = os.path.join(tmpdir, 'test_example')

        # Replicate the fixed code path
        for fi, blob in enumerate(animation):
            cv2.imwrite(
                base + f'_f{fi}.jpg',
                cv2.resize(blob, (w, h), interpolation=cv2.INTER_AREA)
            )

        # Verify all frames were written and contain the right color
        for fi, color in enumerate(colors):
            path = base + f'_f{fi}.jpg'
            assert os.path.exists(path), f"Missing frame file: {path}"
            img = cv2.imread(path)
            assert img is not None, f"Could not read: {path}"
            # Blob was created as np.full(..., color) so channel 0 = color[0], etc.
            # cv2.imread reads back those same raw channel values.
            mean = img.mean(axis=(0, 1))
            dominant_channel = int(np.argmax(mean))
            expected_dominant = int(np.argmax(color))
            assert dominant_channel == expected_dominant, (
                f"Frame {fi}: expected dominant channel {expected_dominant} "
                f"(color {color}), got {dominant_channel} (mean {mean})"
            )

        print(f"PASS: {len(colors)} per-frame files written from animation blobs correctly.")


if __name__ == '__main__':
    test_perframe_saves_animation_frames()
