import cv2
import cvlib as cv
import argparse
import os


def read_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    image = cv2.imread(path)

    if image is None:
        raise ValueError(f"Could not read image: {path}")

    return image


def find_faces(image):
    faces, _ = cv.detect_face(image)
    return faces


def draw_face_box(image, box, padding=20):
    x1, y1, x2, y2 = box
    height, width = image.shape[:2]

    left = max(0, x1 - padding)
    top = max(0, y1 - padding)
    right = min(width - 1, x2 + padding)
    bottom = min(height - 1, y2 + padding)

    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

    label_y = top - 10 if top > 25 else top + 25

    cv2.putText(
        image,
        "Face",
        (left, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    return image


def process_image(input_path, output_path=None, show=True, padding=20):
    try:
        image = read_image(input_path)
    except Exception as error:
        print(f"Error: {error}")
        return None

    faces = find_faces(image)

    if not faces:
        print("No faces found.")
        return image

    for face in faces:
        draw_face_box(image, face, padding)

    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Saved to {output_path}")

    if show:
        cv2.imshow("Face Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image


def process_video(source="0", output_path=None, padding=20):
    camera_source = int(source) if source.isdigit() else source
    video = cv2.VideoCapture(camera_source)

    if not video.isOpened():
        print(f"Could not open video source: {source}")
        return

    writer = None

    if output_path:
        fps = video.get(cv2.CAP_PROP_FPS) or 30
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, codec, fps, (width, height))

    print("Press q to quit. Press s to save a screenshot.")

    while True:
        ok, frame = video.read()

        if not ok:
            break

        faces = find_faces(frame)

        for face in faces:
            draw_face_box(frame, face, padding)

        if writer:
            writer.write(frame)

        cv2.imshow("Face Detection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s"):
            cv2.imwrite("screenshot.png", frame)
            print("Saved screenshot.png")

    video.release()

    if writer:
        writer.release()

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Detect faces in images or video streams.")

    parser.add_argument(
        "-m",
        "--mode",
        choices=["image", "video"],
        default="image"
    )

    parser.add_argument(
        "-i",
        "--input",
        default="person.png"
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None
    )

    parser.add_argument(
        "-p",
        "--padding",
        type=int,
        default=20
    )

    parser.add_argument(
        "--no-show",
        action="store_true"
    )

    args = parser.parse_args()

    if args.mode == "image":
        process_image(
            input_path=args.input,
            output_path=args.output,
            show=not args.no_show,
            padding=args.padding
        )
    else:
        process_video(
            source=args.input,
            output_path=args.output,
            padding=args.padding
        )


if __name__ == "__main__":
    main()
