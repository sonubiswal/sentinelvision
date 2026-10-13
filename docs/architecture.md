# SentinelVision — Architecture

Companion to the static component diagram in [`../README.md`](../README.md). This file
documents the **dynamic** behaviour: what actually happens, in order, to one frame and to one
violation.

## 1. One frame, end to end

```mermaid
sequenceDiagram
    autonumber
    participant Cam as Camera / file
    participant Ing as Ingestor
    participant Det as Detector
    participant Trk as ByteTrack
    participant Asc as Associator
    participant Pol as Policy (zones)
    participant Vot as Temporal voter
    participant Bus as EventBus
    participant Bus4 as Audit log + snapshots

    Cam->>Ing: frame (BGR ndarray)
    Ing->>Det: frame
    Det->>Trk: model.track(frame, tracker="bytetrack.yaml", persist=True)
    Trk-->>Det: boxes + cls + conf + track IDs
    Det->>Det: per-class threshold filter
    Note over Det: person ≥ 0.35 · positive PPE ≥ 0.50<br/>NO-* violations ≥ 0.65
    Det-->>Asc: Detection[] (track_id populated)

    Asc->>Asc: bind PPE → person (centre / IoU ≥ 0.30 / containment ≥ 0.40)
    Asc->>Asc: classify each slot PRESENT · VIOLATION · UNKNOWN
    Asc-->>Pol: PersonAssociation (per worker)

    Pol->>Pol: required_slots(worker_box) — Phase 3 geofence
    Pol-->>Vot: observed violations ∩ required PPE

    Vot->>Vot: history.append(frame) — deque(maxlen=N)
    Vot->>Vot: commit when ≥ K of last N agree
    Vot->>Vot: resolve only after C consecutive clean frames
    Vot-->>Bus: ViolationEvent (track-bound, content-addressed)

    Bus-->>Bus4: publish to every subscriber
    Note over Bus: in-process queue now<br/>Redis Streams in Phase 4
```

## 2. Why a PPE slot has three states

```mermaid
stateDiagram-v2
    [*] --> UNKNOWN
    UNKNOWN --> PRESENT: positive PPE detection bound
    UNKNOWN --> VIOLATION: NO-* detection bound
    PRESENT --> VIOLATION: PPE lost
    VIOLATION --> PRESENT: PPE reappears
    PRESENT --> UNKNOWN: person occluded / blurred
    VIOLATION --> UNKNOWN: person occluded / blurred
    note right of UNKNOWN
        Never counts as a violation.
        Collapsing this into VIOLATION is
        the #1 false-positive source in
        detection-only PPE repos.
    end note
```

## 3. Violation lifecycle (commit → resolve)

```mermaid
stateDiagram-v2
    [*] --> Observing
    Observing --> Observing: votes < K
    Observing --> Committed: votes ≥ K
    Committed --> Committed: violation still observed
    Committed --> Committed: clean frame, streak < C
    Committed --> Observing: C consecutive clean frames
    note right of Committed
        Emitted exactly once, on the frame
        the vote crosses K. Stale window
        evidence is purged on resolve, so a
        resolved violation cannot re-commit
        from history alone.
    end note
```

## 4. Module dependencies (no cycles)

```mermaid
flowchart LR
    config --> classes
    classes --> detector
    detector --> tracking
    tracking --> association
    association --> worker
    worker --> voting
    voting --> pipeline
    association --> pipeline
    tracking --> events
    events --> pipeline
    visualize --> cli
    pipeline --> cli
```

Arrows point from dependency to dependent. `config` and `classes` sit at the bottom and import
nothing from the package; `pipeline` is the only module that knows about all the others.

## 5. Threading and back-pressure

The Phase 2 pipeline is single-threaded by design: one frame in, one `FrameResult` out. That keeps
the state machine deterministic and the tests weight-free. Concurrency arrives in Phase 4 with one
inference thread per stream, each owning its own `FramePipeline` and sharing one `EventBus`.

The bus is already safe for that: every subscriber has its own `queue.Queue`, `publish` is
non-blocking, and a full subscriber queue **drops and counts** rather than stalling inference.
Dropping a dashboard frame is survivable; stalling the detector is not.

## 6. Extension seams

| Seam | Phase 2 implementation | Later |
|---|---|---|
| `Policy` | `GlobalPolicy` (one site-wide rule) | Phase 3 polygon geofences, per-zone PPE |
| `EventBus` | in-process `queue.Queue` | Phase 4 Redis Streams |
| `Detector` backend | ultralytics `.pt` | Phase 7 ONNX Runtime / TensorRT |
| `clock` | `time.time` | injectable for deterministic replay |
| `detector` | `Detector` wrapper | any object with `names` / `detect` / `track` |
