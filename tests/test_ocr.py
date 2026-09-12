import numpy as np

from furti_ai.ocr import TextDetector


class PaddleThreeCompatibilityStub:
    """Simulate a 3.x wrapper whose deprecated ocr method rejects cls."""

    def predict(self, image):
        raise RuntimeError("predict unavailable in this stub")

    def ocr(self, image, **kwargs):
        if "cls" in kwargs:
            raise TypeError("predict() got an unexpected keyword argument 'cls'")
        return [
            {
                "rec_texts": ["Export"],
                "rec_scores": [0.99],
                "rec_polys": [[[1, 2], [9, 2], [9, 8], [1, 8]]],
            }
        ]


def test_paddleocr_three_compatibility_path_does_not_pass_cls():
    detector = TextDetector(enabled=False)
    detector._ocr = PaddleThreeCompatibilityStub()
    detector.available = True

    lines = detector.detect(np.zeros((20, 20, 3), dtype=np.uint8))

    assert [line.text for line in lines] == ["Export"]
    assert lines[0].bbox.center == (5, 5)


def test_ocr_rescales_boxes_after_reduced_resolution_detection():
    detector = TextDetector(enabled=False, max_dim=320)
    detector._ocr = PaddleThreeCompatibilityStub()
    detector.available = True

    lines = detector.detect(np.zeros((320, 640, 3), dtype=np.uint8))

    assert lines[0].bbox.x == 2
    assert lines[0].bbox.y == 4
    assert lines[0].bbox.width == 16
    assert lines[0].bbox.height == 12
