import pytest
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_facial_landmarks_prediction as rflp

class MockRect:
    def __init__(self, l, t, r, b):
        self._left = l
        self._top = t
        self._right = r
        self._bottom = b

    def left(self): return self._left
    def top(self): return self._top
    def right(self): return self._right
    def bottom(self): return self._bottom

class MockPart:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class MockShape:
    def part(self, idx):
        return MockPart(idx, idx * 2)

@patch('run_facial_landmarks_prediction.get_frontal_face_detector')
@patch('run_facial_landmarks_prediction.shape_predictor')
@patch('run_facial_landmarks_prediction.rectangle')
def test_ComputeFaceLandMarks(mock_rectangle, mock_shape_predictor, mock_get_frontal_face_detector):
    """
    Test ComputeFaceLandMarks successfully detects a face and predicts 68 facial
    landmarks. Uses a synthetic image and mocks dlib's face detector and shape predictor.
    """
    # Create a dummy image
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    # Mock detector to return one face bounding box
    mock_detector_instance = MagicMock()
    mock_detector_instance.return_value = [MockRect(10, 10, 50, 50)]
    mock_get_frontal_face_detector.return_value = mock_detector_instance

    # Mock predictor to return dummy landmarks
    mock_predictor_instance = MagicMock()
    mock_predictor_instance.return_value = MockShape()
    mock_shape_predictor.return_value = mock_predictor_instance

    # Mock rectangle to just pass through scaling logic simply
    mock_rectangle.side_effect = lambda left, top, right, bottom: MockRect(left, top, right, bottom)

    shape, boundingBox = rflp.ComputeFaceLandMarks(img, 'MEE')

    # Verify shape dimensions
    assert shape.shape == (68, 2)
    assert shape[0, 0] == 1 # 0 <= 0 gets replaced by 1 in code
    assert shape[0, 1] == 1 # 0 <= 0 gets replaced by 1
    assert shape[1, 0] == 1
    assert shape[1, 1] == 2
    assert shape[67, 0] == 67
    assert shape[67, 1] == 134

    # Verify bounding box (adjusted by ScalingFactor of 100/200 = 0.5)
    # rect is (10, 10, 50, 50) in small image space
    # Scale back to original? The code says:
    # boundingBox = [rect.left() * SF, rect.top() * SF, width, height]
    # SF = 100 / 200 = 0.5
    # So 10 * 0.5 = 5, 50 * 0.5 = 25
    assert boundingBox == [5, 5, 20, 20]

def test_DrawResults():
    """
    Test DrawResults effectively modifies the input image by drawing 68
    facial landmarks as circles without crashing.
    """
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    shape = np.zeros((68, 2), dtype=int)
    for i in range(68):
        shape[i] = [i, i]

    boundingBox = [10, 10, 40, 40]

    # Execute the draw function (it modifies the image in place)
    rflp.DrawResults(img, shape, boundingBox)

    # Check if a specific point was drawn (BGR: (0,0,255) is red)
    assert img[34, 34, 2] == 255 # Red channel should be 255
    assert img[34, 34, 0] == 0   # Blue channel should be 0

    # Ensure point 0,0 is drawn
    assert img[0, 0, 2] == 255
