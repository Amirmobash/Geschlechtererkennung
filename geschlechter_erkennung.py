"""
Gender Detection from Images or Video Streams
Uses cvlib for face detection and gender classification.
Requirements: opencv-python, cvlib, tensorflow (or tensorflow-cpu)
Install with: pip install opencv-python cvlib tensorflow
"""

import cv2
import cvlib as cv
import numpy as np
import argparse
import sys
import os
from typing import Tuple, List, Optional


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk.

    Args:
        image_path: Path to the image file.

    Returns:
        Image as numpy array (BGR format).

    Raises:
        FileNotFoundError: If the file does not exist or cannot be read.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return image


def detect_faces(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces in an image using cvlib.

    Args:
        image: Input image (BGR).

    Returns:
        List of bounding boxes as (x1, y1, x2, y2).
    """
    faces, confidences = cv.detect_face(image)
    # cvlib returns (x1, y1, x2, y2) already
    return faces


def detect_gender(face_roi: np.ndarray) -> Tuple[str, float]:
    """
    Predict gender from a face region.

    Args:
        face_roi: Cropped face image (BGR).

    Returns:
        Tuple (gender_label, confidence) where gender_label is 'Male' or 'Female'.
    """
    labels, confidences = cv.detect_gender(face_roi)
    # labels: ['male', 'female'], confidences: [prob_male, prob_female]
    idx = np.argmax(confidences)
    gender = 'Male' if labels[idx] == 'male' else 'Female'
    return gender, confidences[idx]


def draw_gender_label(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    gender: str,
    confidence: float,
    padding: int = 20,
    conf_threshold: float = 0.5
) -> np.ndarray:
    """
    Draw a rectangle around the face and put the gender label.

    Args:
        image: Image to draw on (modified in place).
        bbox: Face bounding box (x1, y1, x2, y2).
        gender: 'Male' or 'Female'.
        confidence: Confidence score (0..1).
        padding: Extra pixels around the face for the rectangle.
        conf_threshold: Only draw if confidence >= this value.

    Returns:
        The modified image (same as input).
    """
    if confidence < conf_threshold:
        return image

    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]

    # Apply padding, but stay inside image boundaries
    x1_pad = max(0, x1 - padding)
    y1_pad = max(0, y1 - padding)
    x2_pad = min(w - 1, x2 + padding)
    y2_pad = min(h - 1, y2 + padding)

    # Draw rectangle
    cv2.rectangle(image, (x1_pad, y1_pad), (x2_pad, y2_pad), (0, 255, 0), 2)

    # Prepare label text
    label = f"{gender}: {confidence * 100:.1f}%"
    # Place text above rectangle or inside if not enough space
    text_y = y1_pad - 10 if y1_pad - 10 > 10 else y1_pad + 20
    cv2.putText(
        image, label, (x1_pad, text_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )
    return image


def process_image(
    input_path: str,
    output_path: Optional[str] = None,
    show: bool = True,
    conf_threshold: float = 0.5,
    padding: int = 20
) -> Optional[np.ndarray]:
    """
    Process a single image: detect faces, predict gender, draw results.

    Args:
        input_path: Path to input image.
        output_path: If provided, save the result to this path.
        show: If True, display the image in a window.
        conf_threshold: Minimum confidence to display a prediction.
        padding: Padding around each face.

    Returns:
        Processed image or None if an error occurred.
    """
    try:
        image = load_image(input_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading image: {e}")
        return None

    faces = detect_faces(image)
    if len(faces) == 0:
        print("No faces detected.")
        return image

    for face_bbox in faces:
        x1, y1, x2, y2 = face_bbox
        # Extract face region
        face_roi = image[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue
        gender, confidence = detect_gender(face_roi)
        draw_gender_label(image, face_bbox, gender, confidence, padding, conf_threshold)

    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Result saved to {output_path}")

    if show:
        cv2.imshow("Gender Detection", image)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image


def process_video(
    source: str = "0",
    output_path: Optional[str] = None,
    conf_threshold: float = 0.5,
    padding: int = 20,
    skip_frames: int = 2
) -> None:
    """
    Process a video stream (webcam or file) in real time.

    Args:
        source: Video file path or integer camera index (e.g. "0" for webcam).
        output_path: If provided, save the output video to this path.
        conf_threshold: Minimum confidence to display a prediction.
        padding: Padding around each face.
        skip_frames: Process gender detection only every N frames (for speed).
    """
    # Open video source
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Cannot open video source {source}")
        return

    # Video writer for saving
    writer = None
    if output_path:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    print("Press 'q' to quit, 's' to save current frame as screenshot.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        # Process every 'skip_frames' frame
        if frame_count % skip_frames == 0:
            faces = detect_faces(frame)
            for face_bbox in faces:
                x1, y1, x2, y2 = face_bbox
                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size == 0:
                    continue
                gender, confidence = detect_gender(face_roi)
                draw_gender_label(frame, face_bbox, gender, confidence, padding, conf_threshold)

        if writer:
            writer.write(frame)

        cv2.imshow("Gender Detection - Video", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("screenshot.png", frame)
            print("Screenshot saved as screenshot.png")

        frame_count += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Detect gender from faces in images or video streams."
    )
    parser.add_argument(
        "-m", "--mode", choices=["image", "video"], default="image",
        help="Mode: process a single image or a video stream (default: image)"
    )
    parser.add_argument(
        "-i", "--input", default="person.png",
        help="Input path: image file for 'image' mode, or video file / camera index for 'video' mode (default: person.png)"
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output path for saving the result (image or video). If omitted, no saving."
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.5,
        help="Confidence threshold for displaying predictions (0..1, default: 0.5)"
    )
    parser.add_argument(
        "-p", "--padding", type=int, default=20,
        help="Extra padding (in pixels) around each face (default: 20)"
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not display the result window (useful for batch saving)."
    )
    parser.add_argument(
        "--skip-frames", type=int, default=2,
        help="In video mode, process gender only every N frames (default: 2)."
    )

    args = parser.parse_args()

    if args.mode == "image":
        process_image(
            input_path=args.input,
            output_path=args.output,
            show=not args.no_show,
            conf_threshold=args.threshold,
            padding=args.padding
        )
    else:  # video mode
        process_video(
            source=args.input,
            output_path=args.output,
            conf_threshold=args.threshold,
            padding=args.padding,
            skip_frames=args.skip_frames
        )


if __name__ == "__main__":
    main()