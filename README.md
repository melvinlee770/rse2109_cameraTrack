# Project Reflection: Camera Vision-Based Staging Zone Utilization Monitoring System

## Introduction

This reflection documents my development of a computer vision system designed to monitor and calculate the utilization of staging zones in a logistics company's warehouse. The system employs ArUco fiducial markers for spatial reference and a YOLOv8 object detection model for pallet recognition, culminating in a real-time utilization percentage that supports operational decision-making. Over the course of this project, I progressed through multiple iterative versions, each addressing a specific technical limitation discovered during testing.

## Technical Development and Problem-Solving

The project began with a foundational task: using OpenCV's ArUco detection library to identify four markers placed at the corners of a staging area and computing the enclosed region. An early challenge arose when the system required markers to be oriented in a specific direction. I resolved this by implementing spatial sorting based on coordinate sums and differences, which assigns corner positions (top-left, top-right, bottom-right, bottom-left) regardless of physical marker orientation. For the inner-corner reference mode, I developed an approach that automatically selects the marker corner closest to the zone centroid, eliminating orientation dependency entirely.

Converting pixel measurements to real-world units required establishing a reliable scale. Rather than hardcoding zone dimensions, I leveraged the known physical size of each ArUco tag (2.4 cm × 2.4 cm) to dynamically compute a pixels-per-metre ratio by averaging the detected edge lengths across all visible markers. This approach adapts to varying camera distances and removes the need for manual calibration of zone dimensions.

Camera lens distortion proved to be a significant source of measurement error, particularly for markers near the frame edges. I implemented a calibration pipeline using a ChArUco board, which combines ArUco markers with a checkerboard pattern for robust corner detection. The resulting camera matrix and distortion coefficients are applied to undistort each frame prior to processing, reducing area measurement error from approximately 5–15% to within 1–2%.

## Integrating Object Detection with Zone Tracking

Introducing YOLOv8 pallet detection introduced an unexpected problem: the model frequently misidentified ArUco markers as pallets due to their visually similar dark square patterns. I addressed this through a three-layer filtering mechanism that rejects detections below a minimum bounding box area, removes detections exceeding a 30% overlap ratio with known ArUco marker regions, and enforces a confidence threshold.

A more fundamental issue emerged during utilization testing. Pallets located at the periphery of the staging zone produced inflated bounding boxes because the camera's perspective caused their side faces to be included in the detection. This directly distorted the utilization calculation. My initial approach of warping bounding boxes into a bird's-eye coordinate system improved accuracy, but the key insight was recognising that pallet dimensions are fixed and known. Since the logistics company uses only two standard pallet types — US pallets (1.219 m × 1.016 m) and UK pallets (1.200 m × 0.800 m) — I redesigned the system to use YOLO solely for identifying the pallet type and centre position. The utilization calculation then stamps a fixed-size footprint at the warped centre coordinate, completely bypassing bounding box size distortion.

## System Robustness and Stability

The final development phase focused on production reliability. I implemented a multi-pass ArUco detector that applies multiple preprocessing variants (CLAHE enhancement, sharpening, Gaussian blur) with two differently-tuned detector parameter sets, significantly improving detection rates under challenging lighting. A temporal smoothing algorithm with jump rejection stabilises the zone boundary across frames, and a partial detection mode predicts the fourth corner from three visible markers using the parallelogram rule. The zone lock feature allows operators to permanently fix the staging zone boundary, saving it to a JSON file that is automatically restored on subsequent program starts. A toggleable UI overlay hides detailed metrics while keeping dimension readings visible for operational use.

## Conclusion

This project reinforced the importance of iterative development in applied computer vision. Each version addressed a concrete limitation discovered through physical testing rather than theoretical analysis alone. The most valuable lesson was that domain knowledge — specifically, that pallet dimensions are standardised and loads cannot exceed pallet size — provided a more elegant and accurate solution than purely algorithmic approaches to the perspective distortion problem. The final system demonstrates how combining classical computer vision techniques (ArUco detection, perspective transforms, camera calibration) with deep learning (YOLOv8) and domain-specific constraints can produce a practical, deployable monitoring tool.
