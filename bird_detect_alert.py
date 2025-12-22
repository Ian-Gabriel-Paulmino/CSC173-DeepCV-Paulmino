import cv2
from ultralytics import YOLO
from datetime import datetime
import numpy as np

# ===================== CONFIGURATION =====================
# MODEL_PATH = "./results/bird_detection_with_negatives3/weights/best.pt"
MODEL_PATH = "./default_results/bird_detection_without_negatives/weights/best.pt"
VIDEO_PATH = "actual_airstrike_footage.mp4"
OUTPUT_PATH = "output_result_with_alerts.mp4"
CONFIDENCE_THRESHOLD = 0.50  
# =========================================================


class BirdDetectionAlertSystem:
    def __init__(self, model_path, confidence_threshold=0.5):
        """Initialize the alert system with a YOLO OBB model."""
        print("Loading YOLO11n-OBB model...")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.total_detections = 0

    def process_frame(self, frame):
        """
        Process a single frame and return annotated frame + alert status.

        Returns:
            annotated_frame (ndarray)
            bird_detected (bool)
            num_birds (int)
        """

        # Run inference
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            verbose=False
        )

        detections = results[0]

        # ----------------- OBB FIX -----------------
        bird_count = 0
        if detections.obb is not None:
            # Filter by confidence explicitly
            confs = detections.obb.conf
            bird_count = int((confs >= self.confidence_threshold).sum())
        # -------------------------------------------

        self.total_detections += bird_count

        # Annotate frame (OBB-aware)
        annotated_frame = detections.plot()

        # Display bird count
        if bird_count > 0:
            cv2.putText(
                annotated_frame,
                f"Birds Detected: {bird_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )

        return annotated_frame, bird_count > 0, bird_count

    def process_video(self, video_path, output_path, display=False):
        """Process entire video and save output with alerts."""

        print(f"Opening video: {video_path}")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error: Cannot open video file.")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        print("Video Properties:")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")

        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

        frame_count = 0
        alert_display_frames = 0

        print("Processing video...")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1

            annotated_frame, birds_detected, num_birds = self.process_frame(frame)

            # Trigger alert
            if birds_detected:
                alert_display_frames = 15
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] ALERT! {num_birds} bird(s) detected at frame {frame_count}")

            # Display alert box
            if alert_display_frames > 0:
                cv2.rectangle(annotated_frame, (10, 50), (450, 130), (0, 0, 255), 3)

                cv2.putText(
                    annotated_frame,
                    "WARNING!",
                    (30, 90),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.4,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    annotated_frame,
                    "BIRDS DETECTED IN AIRSPACE",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                alert_display_frames -= 1

            # Frame counter
            cv2.putText(
                annotated_frame,
                f"Frame: {frame_count}",
                (width - 250, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1
            )

            out.write(annotated_frame)

            if display:
                cv2.imshow("Bird Detection Alert System", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if frame_count % 50 == 0:
                print(f"   Processed {frame_count} frames...")

        cap.release()
        out.release()
        if display:
            cv2.destroyAllWindows()

        print("\nProcessing Complete!")
        print(f"   Output saved to: {output_path}")


def main():
    print("=" * 70)
    print("BIRD DETECTION ALERT SYSTEM FOR AVIATION SAFETY")
    print("=" * 70)

    system = BirdDetectionAlertSystem(
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )

    print("\nStarting video processing...")
    print(f"   Input:  {VIDEO_PATH}")
    print(f"   Output: {OUTPUT_PATH}")
    print(f"   Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print()

    system.process_video(
        video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH,
        display=False
    )

    print("\n" + "=" * 70)
    print("All operations completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
