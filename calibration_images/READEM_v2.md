# Weekly Reflection — Warehouse Staging Zone Utilization Project

**Week of:** March 10 – March 16, 2026
**Project:** Staging Zone Area Utilization Tracking System

---

## Summary of Work This Week

This week focused on reviewing the current codebase (Staging Zone Tracker v14) and conducting a detailed technical analysis to identify its limitations and plan the next phase of development. The primary concern raised by the team was that the current single-camera system underestimates utilization when stacked pallets occlude nearby single pallets from the camera's view. The majority of the week was spent evaluating architectural approaches to solve this problem and planning a multi-camera, multi-object tracking system.

---

## Current System Analysis (v14)

A thorough code review of `custom_track_v14.py` was completed. The current system performs the following:

- Detects the staging zone boundary using four ArUco markers with a robust multi-pass detection strategy (multiple preprocessing variants and detector configurations).
- Tracks zone stability using exponential smoothing and rolling averages, with support for partial detection (3 out of 4 markers) through parallelogram-based corner prediction.
- Detects pallets using a custom-trained YOLO model (`best_v4.pt`) and classifies them by type (US: 48×40 in / UK: 1200×800 mm).
- Calculates floor area utilization by warping the camera view into a bird's-eye perspective and stamping fixed real-world pallet footprints onto an occupancy mask, rather than relying on pixel-level bounding box area.
- Supports zone locking, saving/loading zone data to JSON, real-time UI overlay, and configurable parameters.

The core utilization math is sound — the fixed-footprint approach with perspective correction gives accurate results when pallets are visible.

---

## Key Problem Identified

The team identified a critical accuracy issue: **tall stacked pallets block the camera's line of sight to adjacent single pallets on the floor**. These occluded pallets are never detected by YOLO, so the system does not count the floor area they occupy. This leads to underestimation of the actual staging area utilization.

This is not a model accuracy issue — the YOLO model performs well on visible pallets. It is a fundamental limitation of single-viewpoint camera-based detection.

---

## Proposed Solution Architecture

After evaluating multiple approaches, the following multi-layered solution was identified as the best fit for the project's constraints (two-camera setup, adjacent zones, 30-minute reporting cycle).

### Layer 1: Extended Detection Boundary

Expand the detection region approximately 1–1.5 meters into the surrounding forklift/worker aisles beyond the ArUco-defined staging zone. This ensures every pallet is detected and assigned an ID while still in an unoccluded aisle space before being placed into the zone.

### Layer 2: Multi-Camera Tracking with Cross-Camera Handoff

Deploy two cameras covering two adjacent staging zones with an overlapping field of view along the shared middle boundary. Key components include:

- Per-camera object tracking using Ultralytics' built-in BoT-SORT/ByteTrack via `model.track()`.
- A shared floor-plane coordinate system anchored by ArUco markers visible to both cameras, enabling projection of all detections into common real-world meter coordinates.
- A global coordinator that merges track IDs when detections from both cameras match by floor proximity within the overlap zone.

### Layer 3: Occlusion Memory with Persistent Pallet IDs

This is the layer that directly addresses the core occlusion problem. Each tracked pallet maintains a state:

- **Visible** — currently detected by at least one camera.
- **Occluded-Presumed-Present** — detections stopped, but no exit trajectory was observed. The pallet is assumed to still be at its last known floor position and continues to count toward utilization.
- **Exited** — the pallet was tracked moving toward and through an aisle boundary, indicating it was physically removed from the zone.

Ghost (occluded) pallets are reconciled using evidence gathered throughout the 30-minute reporting window:

- If the occluding stack moves and the camera sees the pallet again, the ghost is confirmed.
- If the occluding stack moves and the pallet is not there, the ghost is retired.
- If a matching pallet type appears in the other zone (entered from the connecting aisle), the ghost is reassigned rather than double-counted.
- If no evidence is available at all, time-based decay rules and optional manual verification alerts are used as fallbacks.

---

## Risk Identified: Cross-Zone Pallet Movement

A specific risk was identified with the occlusion memory approach. If a worker moves an occluded pallet from Zone A to Zone B while it is in a camera blind spot, the system could double-count it — the ghost persists in Zone A while a new detection appears in Zone B.

The mitigation is to implement a global ghost registry check: before assigning any new ID to an unmatched detection, the coordinator checks all active ghost pallets across all zones. If a ghost matches by pallet type, timing, and entry direction, the new detection inherits the ghost's ID instead of receiving a fresh one. This turns the double-counting risk into a self-correcting mechanism.

---

## Risk Identified: Complete Information Blackout

A scenario was identified where the system has zero evidence about a ghost pallet — the occlusion persists, no matching pallet appears elsewhere, and the camera never regains visibility of that spot. Three mitigation strategies were proposed:

1. **Time-based decay** — Ghost pallets are retired after a configurable maximum lifetime (e.g., 90 minutes or 3 reporting cycles) if never re-confirmed.
2. **Manual verification alerts** — Long-duration ghosts trigger notifications to warehouse staff for a quick physical check.
3. **Forklift proximity detection** — If no forklift or worker activity is detected near the ghost's area, the pallet almost certainly hasn't moved, giving higher confidence to keep counting it.

---

## YOLO ArUco Model Integration

An additional idea was explored: using a separately trained YOLO model for ArUco marker detection alongside the existing OpenCV-based detection. The conclusion was that they can complement each other effectively — YOLO serves as a region proposer to locate markers under difficult conditions (blur, shadows, partial occlusion), and OpenCV extracts precise sub-pixel corner geometry from those regions. This two-stage pipeline could improve zone detection stability, which in turn improves the accuracy of the utilization calculation. This remains a secondary priority behind the multi-camera tracking work.

---

## Next Steps

1. Integrate Ultralytics built-in tracking (`model.track()`) into the existing per-camera loop to establish persistent pallet IDs within each camera view.
2. Set up the shared floor coordinate system using ArUco markers along the shared zone boundary and implement the cross-camera ID coordinator.
3. Implement the pallet state machine (visible / occluded-presumed-present / exited) with ghost pallet logic feeding into the existing `calc_util_fixed()` function.
4. Define and tune reconciliation rules for the 30-minute reporting cycle.
5. Test cross-zone pallet movement scenarios to validate the ghost registry matching logic.

---

## Reflection

This week was primarily analytical and architectural. No code was committed, but the groundwork laid is critical — the occlusion problem cannot be solved by improving the YOLO model or tweaking detection parameters alone. It requires a fundamental shift from per-frame detection to persistent state tracking. The 30-minute reporting window is a significant advantage that simplifies the design, as the system doesn't need to be correct at every instant — only at each reporting checkpoint. The camera layout (adjacent zones with surrounding aisles) also works in our favour, since pallets must pass through visible space to enter or leave, guaranteeing initial detection in almost all cases.

The main technical risk going forward is the complexity of the coordinator — managing global IDs, ghost states, and cross-camera merging across two threaded camera loops introduces concurrency and state management challenges that the current single-camera monolithic loop does not have. Careful architecture of the shared state and message passing between camera threads and the coordinator will be important.