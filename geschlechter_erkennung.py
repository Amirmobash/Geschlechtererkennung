import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import cvlib as cv


Box = Tuple[int, int, int, int]


@dataclass
class FaceDetection:
    box: Box
    confidence: float = 0.0


@dataclass
class AppConfig:
    mode: str
    input_path: str
    output_path: Optional[str]
    padding: int
    show_window: bool
    screenshot_dir: str
    min_confidence: float
    box_color: Tuple[int, int, int]
    text_color: Tuple[int, int, int]


class FaceDetectionApp:
    def __init__(self, config: AppConfig):
        self.config = config
        self.window_name = "Face Detection"
        self.frame_count = 0
        self.last_time = time.time()
        self.fps = 0.0

    def run(self) -> None:
        if self.config.mode == "image":
            self.process_image()
        else:
            self.process_video()

    def process_image(self) -> Optional[cv2.Mat]:
        image = self.load_image(self.config.input_path)
        detections = self.detect_faces(image)

        if not detections:
            print("No faces found.")
        else:
            for detection in detections:
                self.draw_detection(image, detection)

            print(f"Detected {len(detections)} face(s).")

        if self.config.output_path:
            self.save_image(image, self.config.output_path)

        if self.config.show_window:
            self.show_image(image)

        return image

    def process_video(self) -> None:
        source = self.parse_video_source(self.config.input_path)
        video = cv2.VideoCapture(source)

        if not video.isOpened():
            print(f"Could not open video source: {self.config.input_path}")
            return

        writer = self.create_video_writer(video) if self.config.output_path else None

        print("Press q to quit.")
        print("Press s to save a screenshot.")

        try:
            while True:
                ok, frame = video.read()

                if not ok:
                    break

                self.update_fps()

                detections = self.detect_faces(frame)

                for detection in detections:
                    self.draw_detection(frame, detection)

                self.draw_status(frame, len(detections))

                if writer:
                    writer.write(frame)

                cv2.imshow(self.window_name, frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                if key == ord("s"):
                    self.save_screenshot(frame)

        finally:
            video.release()

            if writer:
                writer.release()

            cv2.destroyAllWindows()

    def load_image(self, path: str) -> cv2.Mat:
        image_path = Path(path)

        if not image_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(f"Could not read image: {path}")

        return image

    def detect_faces(self, image: cv2.Mat) -> List[FaceDetection]:
        boxes, confidences = cv.detect_face(image)

        detections = []

        for index, box in enumerate(boxes):
            confidence = float(confidences[index]) if index < len(confidences) else 0.0

            if confidence < self.config.min_confidence:
                continue

            detections.append(
                FaceDetection(
                    box=tuple(map(int, box)),
                    confidence=confidence
                )
            )

        return detections

    def draw_detection(self, image: cv2.Mat, detection: FaceDetection) -> None:
        left, top, right, bottom = self.expand_box(
            detection.box,
            image.shape[1],
            image.shape[0],
            self.config.padding
        )

        cv2.rectangle(
            image,
            (left, top),
            (right, bottom),
            self.config.box_color,
            2
        )

        label = f"Face {detection.confidence:.2f}"

        text_size, _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            2
        )

        text_width, text_height = text_size
        label_top = max(0, top - text_height - 12)
        label_bottom = label_top + text_height + 10

        cv2.rectangle(
            image,
            (left, label_top),
            (left + text_width + 12, label_bottom),
            self.config.box_color,
            -1
        )

        cv2.putText(
            image,
            label,
            (left + 6, label_bottom - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            self.config.text_color,
            2
        )

    def draw_status(self, frame: cv2.Mat, face_count: int) -> None:
        text = f"Faces: {face_count} | FPS: {self.fps:.1f}"

        cv2.rectangle(
            frame,
            (10, 10),
            (260, 45),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    def expand_box(self, box: Box, width: int, height: int, padding: int) -> Box:
        x1, y1, x2, y2 = box

        left = max(0, x1 - padding)
        top = max(0, y1 - padding)
        right = min(width - 1, x2 + padding)
        bottom = min(height - 1, y2 + padding)

        return left, top, right, bottom

    def show_image(self, image: cv2.Mat) -> None:
        cv2.imshow(self.window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_image(self, image: cv2.Mat, path: str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        saved = cv2.imwrite(str(output_path), image)

        if saved:
            print(f"Saved to {output_path}")
        else:
            print(f"Could not save image to {output_path}")

    def save_screenshot(self, frame: cv2.Mat) -> None:
        screenshot_dir = Path(self.config.screenshot_dir)
        screenshot_dir.mkdir(parents=True, exist_ok=True)

        filename = time.strftime("screenshot_%Y%m%d_%H%M%S.png")
        path = screenshot_dir / filename

        saved = cv2.imwrite(str(path), frame)

        if saved:
            print(f"Saved {path}")
        else:
            print("Could not save screenshot.")

    def create_video_writer(self, video: cv2.VideoCapture) -> cv2.VideoWriter:
        fps = video.get(cv2.CAP_PROP_FPS)

        if not fps or fps <= 1:
            fps = 30.0

        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        codec = cv2.VideoWriter_fourcc(*"mp4v")

        return cv2.VideoWriter(
            str(output_path),
            codec,
            fps,
            (width, height)
        )

    def parse_video_source(self, source: str) -> Union[int, str]:
        return int(source) if source.isdigit() else source

    def update_fps(self) -> None:
        self.frame_count += 1
        now = time.time()
        elapsed = now - self.last_time

        if elapsed >= 0.5:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_time = now


def build_config() -> AppConfig:
    parser = argparse.ArgumentParser(
        prog="face_detector",
        description="Detect faces in images, videos, webcams, or stream sources."
    )

    parser.add_argument(
        "-m",
        "--mode",
        choices=["image", "video"],
        default="image",
        help="Choose image or video mode."
    )

    parser.add_argument(
        "-i",
        "--input",
        default="person.png",
        help="Image path, video path, webcam number, or stream URL."
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional output image or video path."
    )

    parser.add_argument(
        "-p",
        "--padding",
        type=int,
        default=20,
        help="Extra space around each detected face."
    )

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum face detection confidence."
    )

    parser.add_argument(
        "--screenshot-dir",
        default="screenshots",
        help="Folder for saved screenshots in video mode."
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a preview window in image mode."
    )

    args = parser.parse_args()

    return AppConfig(
        mode=args.mode,
        input_path=args.input,
        output_path=args.output,
        padding=max(0, args.padding),
        show_window=not args.no_show,
        screenshot_dir=args.screenshot_dir,
        min_confidence=max(0.0, min(1.0, args.min_confidence)),
        box_color=(0, 255, 0),
        text_color=(0, 0, 0)
    )


def main() -> None:
    try:
        config = build_config()
        app = FaceDetectionApp(config)
        app.run()
    except KeyboardInterrupt:
        print("Stopped by user.")
    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
```
