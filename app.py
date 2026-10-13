"""SentinelVision — live demo (Phase 2: association · temporal voting · audit chain).

Run:  python app.py
Env:  SV_MODEL=models/ppe.pt   path to the weights (default shown)

What this demo does
  * Image tab  - detect, bind PPE to workers (IoU / containment), apply your policy.
  * Video tab  - same pipeline over footage, with a light IoU tracker and the
                 N-of-M temporal vote (default 6 of 10) before a violation is committed.
  * Audit tab  - every committed event is SHA-256 hashed into an append-only,
                 hash-chained JSONL log. Edit a line, press Verify, watch it break.
  * Policy tab - the ten-class taxonomy with per-class confidence bars.

Note: on Gradio >= 6, move `theme=`, `css=` and `js=` from gr.Blocks(...) to demo.launch(...).
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from html import escape as esc
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from sentinelvision.detector import Detector

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
MODEL_PATH = Path(os.environ.get("SV_MODEL", "models/ppe.pt"))
_detector = None


def get_detector() -> Detector:
    global _detector
    if _detector is None:
        if not MODEL_PATH.exists():
            raise gr.Error(
                f"Model weights not found at '{MODEL_PATH}'. "
                "Place your PPE weights there or set SV_MODEL to their path."
            )
        _detector = Detector.load(str(MODEL_PATH))
    return _detector


# --------------------------------------------------------------------------- #
# Taxonomy & policy  (mirrors the landing page: 10 classes, asymmetric thresholds)
# --------------------------------------------------------------------------- #
TAXONOMY = [  # (class name, confidence bar, family)
    ("person", 0.35, "ctx"),
    ("Hardhat", 0.50, "pos"),
    ("NO-Hardhat", 0.65, "vio"),
    ("Safety Vest", 0.50, "pos"),
    ("NO-Safety Vest", 0.65, "vio"),
    ("Mask", 0.50, "pos"),
    ("NO-Mask", 0.65, "vio"),
    ("Safety Cone", 0.55, "ctx"),
    ("machinery", 0.55, "ctx"),
    ("vehicle", 0.55, "ctx"),
]


def _slug(name: str) -> str:
    return str(name).strip().lower().replace(" ", "-").replace("_", "-")


THRESH = {_slug(n): t for n, t, _ in TAXONOMY}
FAMILY = {_slug(n): f for n, _, f in TAXONOMY}
PPE_ITEMS = {"hardhat": "Hardhat", "safety-vest": "Safety Vest", "mask": "Mask"}
DEFAULT_REQUIRED = ["hardhat", "safety-vest"]

WINDOW_DEFAULT = 10
QUORUM_DEFAULT = 6
GENESIS = "0" * 64


def _family(name: str) -> str:
    s = _slug(name)
    if s in FAMILY:
        return FAMILY[s]
    if s.startswith("no-") and s[3:] in PPE_ITEMS:
        return "vio"
    return "ctx"


def _item_of(slug: str, fam: str) -> str:
    return slug[3:] if fam == "vio" else slug


def _cls_name(item: str, state: str) -> str:
    base = PPE_ITEMS.get(item, item)
    return f"NO-{base}" if state == "absent" else base


# --------------------------------------------------------------------------- #
# Palette (BGR for OpenCV, taken from the site's tokens)
# --------------------------------------------------------------------------- #
def _hex_bgr(h: str):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


C = {
    "pos": _hex_bgr("#12a56b"),
    "vio": _hex_bgr("#e5384f"),
    "ctx": _hex_bgr("#8d9bb0"),
    "id": _hex_bgr("#2457ff"),
    "dk": _hex_bgr("#0b1623"),
    "or": _hex_bgr("#ff5a1f"),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX
STATUS_LABEL = {"violation": "VIOLATION", "compliant": "COMPLIANT", "unverified": "UNVERIFIED"}
STATUS_FAMILY = {"violation": "vio", "compliant": "pos", "unverified": "id"}


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def _iou(a, b):
    i = _inter(a, b)
    u = _area(a) + _area(b) - i
    return i / u if u > 0 else 0.0


def _center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _inside(pt, b):
    return b[0] <= pt[0] <= b[2] and b[1] <= pt[1] <= b[3]


# --------------------------------------------------------------------------- #
# Detect -> associate -> judge
# --------------------------------------------------------------------------- #
def run_detector(bgr: np.ndarray, floor: float, strict: bool):
    """Run the model and normalise detections; optionally apply per-class bars."""
    raw = get_detector().detect(bgr, conf=float(floor))
    out = []
    for d in raw:
        name = d.raw_name
        det = {
            "name": name,
            "slug": _slug(name),
            "conf": float(d.confidence),
            "box": tuple(float(v) for v in d.xyxy),
            "track_id": getattr(d, "track_id", None),
        }
        if strict and det["conf"] < THRESH.get(det["slug"], 0.55):
            continue
        out.append(det)
    return out


def _best_worker(d, workers):
    """Deterministic binding: containment first, then overlap share, then smaller person."""
    c = _center(d["box"])
    a = _area(d["box"]) or 1.0
    best, best_key = None, None
    for w in workers:
        inside = _inside(c, w["box"])
        if not inside and _iou(d["box"], w["box"]) < 0.05:
            continue
        key = (inside, round(_inter(d["box"], w["box"]) / a, 4), -_area(w["box"]), -w["idx"])
        if best_key is None or key > best_key:
            best, best_key = w, key
    return best


def associate(dets):
    persons = sorted((d for d in dets if d["slug"] == "person"), key=lambda d: (d["box"][0], d["box"][1]))
    workers = []
    for i, p in enumerate(persons):
        p["worker"] = i + 1
        workers.append(
            {"idx": i + 1, "box": p["box"], "conf": p["conf"], "track_id": p["track_id"], "items": {}}
        )
    context, leftovers = [], []
    for d in dets:
        if d["slug"] == "person":
            continue
        fam = _family(d["name"])
        if fam == "ctx":
            context.append(d)
            continue
        item = _item_of(d["slug"], fam)
        w = _best_worker(d, workers)
        if w is None:
            leftovers.append(d)
            continue
        state = "absent" if fam == "vio" else "present"
        cur = w["items"].get(item)
        if (
            cur is None
            or d["conf"] > cur["conf"]
            or (d["conf"] == cur["conf"] and state == "present")  # tie -> favour compliance
        ):
            w["items"][item] = {"state": state, "conf": d["conf"], "box": d["box"]}
        d["worker"] = w["idx"]
    return workers, context, leftovers


def judge(worker, required):
    ok, missing, unk = [], [], []
    for item in required:
        st = worker["items"].get(item)
        if st is None:
            unk.append(item)
        elif st["state"] == "absent":
            missing.append(item)
        else:
            ok.append(item)
    status = "violation" if missing else ("unverified" if unk else "compliant")
    return {"status": status, "ok": ok, "missing": missing, "unverified": unk}


def _wid(w) -> str:
    return f"#{w['track_id']}" if w.get("track_id") is not None else f"W{w['idx']}"


class IoUTracker:
    """Tiny greedy IoU tracker, used only when the detector gives no track ids."""

    def __init__(self, thr: float = 0.3, max_age: int = 8):
        self.thr, self.max_age, self.next, self.tracks = thr, max_age, 1, {}

    def update(self, workers):
        cand = []
        for wi, w in enumerate(workers):
            if w["track_id"] is not None:
                continue
            for tid, t in self.tracks.items():
                i = _iou(w["box"], t["box"])
                if i >= self.thr:
                    cand.append((i, wi, tid))
        cand.sort(reverse=True)
        used_w, used_t = set(), set()
        for _, wi, tid in cand:
            if wi in used_w or tid in used_t:
                continue
            workers[wi]["track_id"] = tid
            used_w.add(wi)
            used_t.add(tid)
        seen = set()
        for w in workers:
            if w["track_id"] is None:
                w["track_id"] = self.next
                self.next += 1
            self.tracks[w["track_id"]] = {"box": w["box"], "age": 0}
            seen.add(w["track_id"])
        for tid in list(self.tracks):
            if tid not in seen:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def _box(img, b, color, thick):
    cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, thick, cv2.LINE_AA)


def _tag(img, text, b, color, scale, anchor="top"):
    H, W = img.shape[:2]
    (tw, th), bl = cv2.getTextSize(text, FONT, scale, 1)
    pad = 4
    bw, bh = tw + 2 * pad, th + bl + 2 * pad
    x = int(max(0, min(b[0], W - bw)))
    if anchor == "top":
        y0 = int(b[1]) - bh
        if y0 < 0:
            y0 = int(b[1])
    else:
        y0 = int(b[3])
        if y0 + bh > H:
            y0 = int(b[3]) - bh
    cv2.rectangle(img, (x, y0), (x + bw, y0 + bh), color, -1)
    cv2.putText(img, text, (x + pad, y0 + pad + th), FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)


def annotate(bgr, workers, context, leftovers, required, hud):
    out = bgr.copy()
    H, W = out.shape[:2]
    s = min(max(max(H, W) / 900.0, 0.5), 1.2)
    thick = max(2, round(2 * s))

    for d in context + leftovers:
        col = C[_family(d["name"])]
        _box(out, d["box"], col, max(1, thick - 1))
        _tag(out, f"{d['name']} {d['conf']:.2f}", d["box"], col, 0.42 * s + 0.1)

    n_vio = 0
    for w in workers:
        j = judge(w, required)
        n_vio += j["status"] == "violation"
        pcol = C[STATUS_FAMILY[j["status"]]]
        _box(out, w["box"], pcol, thick + 1)
        for item, st in w["items"].items():
            fam = "vio" if st["state"] == "absent" else "pos"
            _box(out, st["box"], C[fam], thick)
            _tag(out, f"{_cls_name(item, st['state'])} {st['conf']:.2f}", st["box"], C[fam], 0.42 * s + 0.1)
        _tag(out, f"{_wid(w)} {STATUS_LABEL[j['status']]}", w["box"], pcol, 0.5 * s + 0.1, anchor="bottom")

    # HUD strip
    bar = int(30 * s) + 6
    ov = out.copy()
    cv2.rectangle(ov, (0, 0), (W, bar), C["dk"], -1)
    out = cv2.addWeighted(ov, 0.78, out, 0.22, 0)
    ty = int(bar * 0.68)
    cv2.putText(out, hud, (10, ty), FONT, 0.5 * s + 0.1, (255, 255, 255), 1, cv2.LINE_AA)
    right = f"{n_vio} VIOLATION{'S' if n_vio != 1 else ''}" if n_vio else "CLEAR"
    rc = (141, 141, 255) if n_vio else (168, 224, 95)
    (tw, _), _ = cv2.getTextSize(right, FONT, 0.5 * s + 0.1, 1)
    cv2.putText(out, right, (W - tw - 10, ty), FONT, 0.5 * s + 0.1, rc, 1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- #
# Tamper-evident audit chain
# --------------------------------------------------------------------------- #
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _digest(rec: dict) -> str:
    body = {k: v for k, v in rec.items() if k != "sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _seal(record: dict, prev: str) -> dict:
    rec = dict(record)
    rec["prev_sha256"] = prev
    rec["sha256"] = _digest(rec)
    return rec


def _dedupe_key(e: dict) -> str:
    return json.dumps(
        [e.get("source"), e.get("worker"), e.get("track_id"),
         [(f.get("class"), f.get("verdict")) for f in e.get("findings", [])]]
    )


def append_events(chain, events):
    chain = list(chain)
    seen = {_dedupe_key(e) for e in chain}
    prev = chain[-1]["sha256"] if chain else GENESIS
    added = 0
    for e in events:
        if _dedupe_key(e) in seen:
            continue
        rec = _seal(e, prev)
        chain.append(rec)
        seen.add(_dedupe_key(rec))
        prev = rec["sha256"]
        added += 1
    return chain, added


def verify_records(records):
    prev = GENESIS
    for i, r in enumerate(records, start=1):
        if r.get("prev_sha256") != prev:
            return False, i, "link to the previous record is broken"
        if _digest(r) != r.get("sha256"):
            return False, i, "record contents no longer match their hash"
        prev = r["sha256"]
    return True, None, ""


def make_event(w, required, meta, votes, window):
    """One worker record. `votes` maps item -> agreeing frames for committed violations."""
    findings = []
    for item in required:
        st = w["items"].get(item)
        if item in votes:
            conf = st["conf"] if st else None
            findings.append({
                "class": _cls_name(item, "absent"),
                "conf": round(conf, 3) if conf is not None else None,
                "verdict": "violation", "votes": votes[item], "window": window,
            })
        elif st and st["state"] == "present":
            findings.append({"class": _cls_name(item, "present"), "conf": round(st["conf"], 3), "verdict": "present"})
    return {
        "ts": _now(),
        "camera": meta["camera"],
        "zone": meta["zone"],
        "source": meta["source"],
        "track_id": w.get("track_id"),
        "worker": w["idx"],
        "findings": findings,
        "policy": meta["policy"],
    }


def _policy_str(zone, required):
    names = "+".join(required) if required else "none"
    return f"{zone}: {names} required"


def _jsonl(chain) -> str:
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in chain)


def _short(h: str) -> str:
    return f"{h[:4]}…{h[-4:]}" if h and len(h) > 10 else h


def badge(ok: bool, text: str) -> str:
    cls = "ok" if ok else "bad"
    return f'<div class="sv-chain {cls}"><i></i>{esc(text)}</div>'


def chain_badge(chain) -> str:
    if not chain:
        return badge(True, "Chain empty · the first event will anchor to the genesis hash")
    ok, _, _ = verify_records(chain)
    head = _short(chain[-1]["sha256"])
    return badge(ok, f"Chain intact · {len(chain)} event{'s' if len(chain) != 1 else ''} · head {head}")


EVENTS_FILE = Path(tempfile.gettempdir()) / "sentinelvision_events.jsonl"


def audit_updates(chain):
    if chain:
        EVENTS_FILE.write_text(_jsonl(chain) + "\n", encoding="utf-8")
        f = str(EVENTS_FILE)
    else:
        f = None
    return _jsonl(chain), f, chain_badge(chain)


def verify_text(text: str) -> str:
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return badge(True, "Nothing to verify yet")
    recs = []
    for n, ln in enumerate(lines, start=1):
        try:
            recs.append(json.loads(ln))
        except json.JSONDecodeError:
            return badge(False, f"Line {n} is not valid JSON")
    ok, idx, why = verify_records(recs)
    if ok:
        return badge(True, f"Verified · {len(recs)} records · head {_short(recs[-1]['sha256'])}")
    return badge(False, f"Tampering detected at record {idx}: {why}")


def clear_chain():
    return [], *audit_updates([])


# --------------------------------------------------------------------------- #
# HTML renderers
# --------------------------------------------------------------------------- #
def _band(cards) -> str:
    h = ""
    for label, value, caption, kind in cards:
        h += (f'<div class="sv-m {kind}"><i>{esc(label)}</i><b>{esc(value)}</b>'
              f'<span>{esc(caption)}</span></div>')
    return f'<div class="sv-band">{h}</div>'


def _worker_card(w, required) -> str:
    j = judge(w, required)
    chips = ""
    for item, label in PPE_ITEMS.items():
        st = w["items"].get(item)
        opt = "" if item in required else " opt"
        if st is None:
            chips += f'<span class="it un{opt}">{esc(label)} · not observed</span>'
        elif st["state"] == "present":
            chips += f'<span class="it ok{opt}">{esc(label)} {st["conf"]:.2f}</span>'
        else:
            chips += f'<span class="it bad{opt}">NO-{esc(label)} {st["conf"]:.2f}</span>'
    return (
        f'<div class="sv-wk {j["status"]}"><div class="h"><b>{esc(_wid(w))}</b>'
        f'<span class="chip">{STATUS_LABEL[j["status"]]}</span>'
        f'<small>person {w["conf"]:.2f}</small></div><div class="its">{chips}</div></div>'
    )


EMPTY_HTML = (
    '<div class="sv-empty"><b>Drop in a site photo.</b>'
    "Each PPE finding will be bound to a worker, judged against your policy, "
    "and any violation is sealed into the audit chain.</div>"
)


def render_image_summary(workers, required, ms, n_dets, added, zone):
    js = [judge(w, required) for w in workers]
    n_v = sum(j["status"] == "violation" for j in js)
    n_c = sum(j["status"] == "compliant" for j in js)
    band = _band([
        ("workers", str(len(workers)), "Persons bound to PPE", ""),
        ("compliant", str(n_c), "All required PPE seen", "ok"),
        ("violations", str(n_v), "Required PPE missing", "bad" if n_v else ""),
        ("latency", f"{ms:.0f} ms", f"{n_dets} detections · detect → judge", ""),
    ])
    pol = " + ".join(PPE_ITEMS[r] for r in required) or "nothing"
    cards = "".join(_worker_card(w, required) for w in workers) or (
        '<div class="sv-empty"><b>No workers found.</b>Lower the detector floor or try a clearer frame.</div>'
    )
    note = (f"{added} new event{'s' if added != 1 else ''} sealed into the audit chain."
            if n_v else "No violations, so nothing was written to the audit chain.")
    return (
        f'{band}<div class="sv-pol">Policy · <b>{esc(zone)}</b> requires <b>{esc(pol)}</b></div>'
        f'<div class="sv-wks">{cards}</div>'
        f'<p class="sv-foot">{esc(note)} A still image counts as 1 of 1 frames; '
        f"use the Video tab for the temporal quorum.</p>"
    )


def render_video_summary(processed, tracks, commits, secs, required, window, quorum):
    band = _band([
        ("frames", str(processed), "Frames analysed", ""),
        ("workers", str(tracks), "Distinct tracks seen", ""),
        ("committed", str(len(commits)), f"Violations passing {quorum} of {window}", "bad" if commits else "ok"),
        ("throughput", f"{processed / secs:.1f} fps" if secs > 0 else "n/a", f"{secs:.1f} s wall time", ""),
    ])
    if commits:
        rows = "".join(
            f'<div class="sv-wk violation"><div class="h"><b>{esc(c["wid"])}</b>'
            f'<span class="chip">COMMITTED</span><small>t = {c["t"]:.1f} s</small></div>'
            f'<div class="its">{"".join(f"""<span class="it bad">{esc(i)}</span>""" for i in c["items"])}</div></div>'
            for c in commits
        )
    else:
        rows = ('<div class="sv-empty"><b>No violation reached quorum.</b>'
                "One bad frame can never raise an event, which is the point.</div>")
    return f'{band}<div class="sv-wks">{rows}</div>'


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #
def _meta(camera, zone, source, required):
    zone = (zone or "zone-1").strip() or "zone-1"
    return {"camera": (camera or "cam-01").strip() or "cam-01", "zone": zone,
            "source": source, "policy": _policy_str(zone, required)}


def run_image(image, required, camera, zone, floor, strict, chain):
    chain = chain or []
    if image is None:
        return (None, EMPTY_HTML, "", chain, *audit_updates(chain))
    required = [r for r in (required or []) if r in PPE_ITEMS]
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # gradio gives RGB, detector expects BGR

    t0 = time.perf_counter()
    dets = run_detector(bgr, floor, strict)
    workers, context, leftovers = associate(dets)
    ms = (time.perf_counter() - t0) * 1000

    meta = _meta(camera, zone, f"image:{hashlib.sha256(image.tobytes()).hexdigest()[:12]}", required)
    events = []
    for w in workers:
        j = judge(w, required)
        if j["status"] == "violation":
            events.append(make_event(w, required, meta, {i: 1 for i in j["missing"]}, 1))
    chain, added = append_events(chain, events)

    hud = f"SENTINELVISION · {meta['camera']} · {meta['zone']} · {len(workers)} workers"
    ann = annotate(bgr, workers, context, leftovers, required, hud)
    rows = [{
        "class": d["name"], "family": _family(d["name"]), "confidence": round(d["conf"], 3),
        "box": [int(v) for v in d["box"]], "worker": d.get("worker"),
    } for d in dets]

    return (
        cv2.cvtColor(ann, cv2.COLOR_BGR2RGB),
        render_image_summary(workers, required, ms, len(dets), added, meta["zone"]),
        json.dumps(rows, indent=2),
        chain,
        *audit_updates(chain),
    )


def run_video(video, required, camera, zone, floor, strict, stride, max_frames, window, quorum,
              chain, progress=gr.Progress()):
    chain = chain or []
    if not video:
        return [], EMPTY_HTML, chain, *audit_updates(chain)
    required = [r for r in (required or []) if r in PPE_ITEMS]
    window, quorum = int(window), min(int(quorum), int(window))
    stride, max_frames = max(1, int(stride)), max(1, int(max_frames))

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise gr.Error("Could not open that video. Try an mp4 / avi / mov file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    expect = min(max_frames, (total // stride) + 1) if total else max_frames

    tracker = IoUTracker()
    hist = defaultdict(lambda: defaultdict(lambda: deque(maxlen=window)))
    lastc = {}
    committed, commits, events, gallery = set(), [], [], []
    src = f"video:{Path(video).name}"
    t0 = time.perf_counter()
    idx, processed = -1, 0

    while processed < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % stride:
            continue
        processed += 1
        progress(min(processed / max(expect, 1), 1.0), desc=f"Frame {processed}/{expect}")

        dets = run_detector(frame, floor, strict)
        workers, context, leftovers = associate(dets)
        tracker.update(workers)
        t_sec = idx / fps

        fired = {}
        for w in workers:
            tid = w["track_id"]
            for item in required:
                st = w["items"].get(item)
                hist[tid][item].append(st["state"] if st else "none")
                if st and st["state"] == "absent":
                    lastc[(tid, item)] = st["conf"]
                votes = sum(1 for s in hist[tid][item] if s == "absent")
                if votes >= quorum and (tid, item) not in committed:
                    committed.add((tid, item))
                    fired.setdefault(tid, (w, {}))[1][item] = votes

        for tid, (w, votes) in fired.items():
            meta = _meta(camera, zone, f"{src}#frame{idx}", required)
            ev = make_event(w, required, meta, votes, window)
            for f in ev["findings"]:  # fall back to the last violating confidence
                if f["verdict"] == "violation" and f["conf"] is None:
                    item = next((k for k in PPE_ITEMS if _cls_name(k, "absent") == f["class"]), None)
                    c = lastc.get((tid, item))
                    f["conf"] = round(c, 3) if c is not None else None
            events.append(ev)
            commits.append({"wid": _wid(w), "t": t_sec, "items": [_cls_name(i, "absent") for i in votes]})
            if len(gallery) < 12:
                hud = f"SENTINELVISION · {meta['camera']} · t={t_sec:.1f}s · COMMIT {_wid(w)}"
                ann = annotate(frame, workers, context, leftovers, required, hud)
                cap_txt = f"{_wid(w)} · {', '.join(commits[-1]['items'])} · t={t_sec:.1f}s"
                gallery.append((cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), cap_txt))
    cap.release()

    chain, _ = append_events(chain, events)
    secs = time.perf_counter() - t0
    summary = render_video_summary(processed, len(hist), commits, secs, required, window, quorum)
    return gallery, summary, chain, *audit_updates(chain)


# --------------------------------------------------------------------------- #
# Static HTML (hero, taxonomy, footer)
# --------------------------------------------------------------------------- #
HERO = """
<div class="sv-hero">
  <div class="sv-brand">
    <svg viewBox="0 0 32 34" fill="none" width="26" height="28"><path d="M16 2l12 6.5v10.8C28 26 20 31 16 33 12 31 4 26 4 19.3V8.5z" fill="#2457ff" stroke="#0d1b2a" stroke-width="2.4" stroke-linejoin="round"/><circle cx="16" cy="17" r="4" fill="#ff5a1f"/></svg>
    SentinelVision <small>EDGE</small>
  </div>
  <span class="sv-pill"><i></i>Phase 2 · tracking · association · temporal voting</span>
  <h1 class="sv-h1">Turn commodity cameras into a <mark>per-worker safety record.</mark></h1>
  <p class="sv-sub">Every PPE observation is bound to a worker, judged against the policy you set, and sealed into a tamper-evident audit trail. Runs on CPU, with no cloud dependency.</p>
  <div class="sv-trust"><span>No cloud dependency</span><span>Realtime on CPU</span><span>Apache-2.0</span></div>
  <div class="sv-pipe">
    <div class="sv-nd"><i>01</i><b>Detect</b><p>YOLOv8 finds persons and PPE, each class gated by its own bar.</p></div>
    <div class="sv-nd"><i>02</i><b>Track</b><p>A stable ID, so findings belong to a worker, not a pixel box.</p></div>
    <div class="sv-nd"><i>03</i><b>Associate</b><p>PPE binds to its worker by containment or IoU, deterministically.</p></div>
    <div class="sv-nd"><i>04</i><b>Vote</b><p>Six of ten frames must agree before a violation is committed.</p></div>
    <div class="sv-nd"><i>05</i><b>Audit</b><p>SHA-256 hash chain in append-only JSONL.</p></div>
  </div>
</div>
"""


def _taxonomy_html() -> str:
    rows = "".join(
        f'<div class="sv-rw {fam}"><i></i><b>{esc(name)}</b>'
        f'<u><s style="width:{thr * 100:.0f}%"></s></u><span>{thr:.2f}</span></div>'
        for name, thr, fam in TAXONOMY
    )
    return (
        '<div class="sv-lg"><span class="pos">Positive PPE</span><span class="vio">Violation</span>'
        f'<span class="ctx">Context class</span></div><div class="sv-cl">{rows}</div>'
        '<div class="sv-note"><b>Asymmetric by design.</b> A violation class sits at 0.65 while its positive '
        "counterpart sits at 0.50, so raising an alarm always demands more evidence than recording compliance. "
        "Combined with the six-of-ten quorum, that ordering keeps a false positive off a supervisor's desk.</div>"
    )


FOOTER = ('<div class="sv-fb"><span>SentinelVision · Apache-2.0</span>'
          "<span>Built with YOLOv8 · ByteTrack · Ultralytics · Pydantic · structlog · Gradio</span></div>")

# --------------------------------------------------------------------------- #
# Theme & CSS (tokens lifted from the landing page)
# --------------------------------------------------------------------------- #
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "-apple-system", "Segoe UI", "Arial", "sans-serif"],
    font_mono=["ui-monospace", "Consolas", "Menlo", "monospace"],
).set(
    body_background_fill="#f3f1ec",
    block_background_fill="#ffffff",
    block_border_color="#0d1b2a",
    block_border_width="2px",
    block_radius="16px",
    button_primary_background_fill="#0d1b2a",
    button_primary_background_fill_hover="#0d1b2a",
    button_primary_text_color="#ffffff",
    button_primary_border_color="#0d1b2a",
)

CSS = """
:root{--bg:#f3f1ec;--ink:#0d1b2a;--mu:#5b6b7d;--ln:#dcd8ce;--bl:#2457ff;--or:#ff5a1f;--ok:#12a56b;--bad:#e5384f;--dk:#0b1623;--mono:ui-monospace,Consolas,Menlo,monospace}
body{background:var(--bg)!important}
body:before{content:"";position:fixed;inset:0;z-index:-1;pointer-events:none;background-image:linear-gradient(var(--ln) 1px,transparent 1px),linear-gradient(90deg,var(--ln) 1px,transparent 1px);background-size:56px 56px;opacity:.45;-webkit-mask-image:radial-gradient(80% 60% at 70% 0,#000,transparent);mask-image:radial-gradient(80% 60% at 70% 0,#000,transparent)}
.gradio-container{background:transparent!important;max-width:min(1760px,100%)!important;width:100%!important;margin:0 auto!important;padding:0 clamp(14px,2.4vw,40px)!important;color:var(--ink)}
.sv-card .block{border-width:1px!important;border-color:var(--ln)!important;box-shadow:none!important}
.sv-card .block.padded{background:transparent!important}
footer{display:none!important}
:focus-visible{outline:3px solid var(--bl)!important;outline-offset:2px}

/* hero */
.sv-hero{padding:22px 4px 6px}
.sv-brand{display:flex;gap:10px;align-items:center;font-weight:800;font-size:18px;letter-spacing:-.03em;margin-bottom:26px}
.sv-brand small{font:700 10px var(--mono);border:1.5px solid var(--ink);border-radius:5px;padding:2px 5px}
.sv-pill{display:inline-flex;gap:9px;align-items:center;font:600 12px var(--mono);color:var(--mu);background:#fff;border:1px solid var(--ln);padding:8px 14px;border-radius:99px}
.sv-pill i{width:8px;height:8px;border-radius:50%;background:var(--ok);animation:svpl 2s infinite}
.sv-h1{font-size:clamp(34px,5.2vw,60px);font-weight:800;letter-spacing:-.04em;line-height:1.05;margin:20px 0 14px;max-width:900px}
.sv-h1 mark{background:linear-gradient(transparent 62%,#ffd9c9 62%);color:inherit}
.sv-sub{color:var(--mu);font-size:17.5px;max-width:680px;line-height:1.6;margin:0}
.sv-trust{display:flex;flex-wrap:wrap;gap:8px 22px;margin-top:18px;font-size:14px;color:var(--mu);font-weight:600}
.sv-trust span:before{content:"✓ ";color:var(--ok);font-weight:800}
.sv-pipe{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:26px 0 8px}
.sv-nd{background:#fff;border:2px solid var(--ink);border-radius:16px;padding:14px;transition:.25s}
.sv-nd:hover{transform:translateY(-4px);box-shadow:5px 5px 0 var(--or)}
.sv-nd i{font:800 12px var(--mono);color:#fff;background:var(--bl);border-radius:8px;padding:3px 8px;font-style:normal;margin-right:8px}
.sv-nd b{font-size:16px;letter-spacing:-.02em}
.sv-nd p{color:var(--mu);font-size:13px;margin:8px 0 0;line-height:1.5}

/* panels & controls */
.sv-card{background:#fff!important;border:2px solid var(--ink)!important;border-radius:20px!important;padding:14px!important;box-shadow:6px 6px 0 var(--ln)}
.sv-side-h{font:700 11px var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--mu);margin:2px 0 4px}
button.sv-go{background:var(--ink)!important;color:#fff!important;border:2px solid var(--ink)!important;border-radius:14px!important;font-weight:700!important;box-shadow:5px 5px 0 var(--or)!important;transition:.25s cubic-bezier(.2,.7,.3,1)!important}
button.sv-go:hover{transform:translate(-3px,-3px);box-shadow:8px 8px 0 var(--or)!important}
button.sv-ghost{background:#fff!important;color:var(--ink)!important;border:2px solid var(--ink)!important;border-radius:14px!important;font-weight:700!important;box-shadow:5px 5px 0 var(--ln)!important;transition:.25s!important}
button.sv-ghost:hover{transform:translate(-3px,-3px);box-shadow:8px 8px 0 var(--bl)!important}
button[role="tab"]{font-weight:700!important}
button[role="tab"].selected{color:var(--bl)!important}
#sv-audit textarea{font:12.5px/1.7 var(--mono)!important;background:#050b13!important;color:#d5e1f1!important;border-radius:14px!important}

/* result band */
.sv-band{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:2px 0 12px}
.sv-m{background:#fff;border:2px solid var(--ink);border-radius:16px;padding:14px;transition:.25s}
.sv-m:hover{transform:translateY(-3px) rotate(-.5deg);box-shadow:5px 5px 0 var(--bl)}
.sv-m i{font:700 10px var(--mono);color:var(--or);text-transform:uppercase;letter-spacing:.14em;font-style:normal}
.sv-m b{display:block;font-size:clamp(22px,2.6vw,32px);letter-spacing:-.045em;line-height:1.1;margin-top:4px}
.sv-m span{display:block;color:var(--mu);font-size:12px;margin-top:6px;line-height:1.35}
.sv-m.ok b{color:var(--ok)}.sv-m.bad b{color:var(--bad)}
.sv-pol{font:600 12.5px var(--mono);color:var(--mu);background:#fff;border:1px dashed var(--ln);border-radius:12px;padding:10px 14px;margin-bottom:12px}
.sv-pol b{color:var(--ink)}

/* worker cards */
.sv-wks{display:grid;gap:10px}
.sv-wk{background:#fff;border:2px solid var(--ink);border-left-width:8px;border-radius:14px;padding:12px 14px;transition:.25s}
.sv-wk:hover{transform:translateX(5px)}
.sv-wk.violation{border-left-color:var(--bad)}.sv-wk.compliant{border-left-color:var(--ok)}.sv-wk.unverified{border-left-color:var(--bl)}
.sv-wk .h{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.sv-wk .h b{font:800 15px var(--mono)}
.sv-wk .h small{margin-left:auto;font:600 11.5px var(--mono);color:var(--mu)}
.chip{font:800 10.5px var(--mono);letter-spacing:.08em;padding:4px 8px;border-radius:6px;color:#fff}
.violation .chip{background:var(--bad)}.compliant .chip{background:var(--ok)}.unverified .chip{background:var(--bl)}
.its{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.it{font:700 11.5px var(--mono);border:1.5px solid currentColor;border-radius:99px;padding:4px 10px}
.it.ok{color:var(--ok)}.it.bad{color:var(--bad)}.it.un{color:var(--mu)}.it.opt{opacity:.55}
.sv-foot{color:var(--mu);font-size:12.5px;margin:12px 2px 0}
.sv-empty{background:#fff;border:2px dashed var(--ln);border-radius:16px;padding:22px;color:var(--mu);font-size:14px;line-height:1.55}
.sv-empty b{display:block;color:var(--ink);font-size:16px;margin-bottom:4px}

/* audit badge */
.sv-chain{display:flex;gap:10px;align-items:center;font:700 12.5px var(--mono);border:2px solid var(--ink);border-radius:12px;padding:10px 14px;background:#fff}
.sv-chain i{width:10px;height:10px;border-radius:50%}
.sv-chain.ok i{background:var(--ok)}.sv-chain.bad{background:#fff0f2;border-color:var(--bad);color:var(--bad)}.sv-chain.bad i{background:var(--bad)}

/* taxonomy */
.sv-lg{display:flex;flex-wrap:wrap;gap:12px 24px;margin:4px 0 14px;font-weight:700;font-size:13.5px}
.sv-lg span:before{content:"";display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:8px;background:var(--c)}
.sv-lg .pos,.sv-rw.pos{--c:var(--ok)}.sv-lg .vio,.sv-rw.vio{--c:var(--bad)}.sv-lg .ctx,.sv-rw.ctx{--c:#8d9bb0}
.sv-cl{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}
.sv-rw{display:grid;grid-template-columns:12px 1fr 90px 44px;gap:12px;align-items:center;background:#fff;border:2px solid var(--ink);border-radius:14px;padding:13px 16px;transition:.25s}
.sv-rw:hover{transform:translateX(6px);box-shadow:-6px 6px 0 var(--c)}
.sv-rw i{width:12px;height:12px;border-radius:50%;background:var(--c)}
.sv-rw b{font:700 14px var(--mono)}
.sv-rw u{height:6px;border-radius:9px;background:var(--ln);text-decoration:none;overflow:hidden}
.sv-rw u s{display:block;height:100%;background:var(--c)}
.sv-rw span{font:800 13px var(--mono);text-align:right}
.sv-note{margin-top:18px;padding:18px 20px;border-radius:16px;background:#fff3ec;border:2px solid var(--or);color:#5a3220;font-size:14.5px;line-height:1.55}
.sv-note b{color:var(--ink)}
.sv-fb{display:flex;flex-wrap:wrap;gap:10px;justify-content:space-between;margin-top:26px;padding:18px 4px;border-top:1px solid var(--ln);color:var(--mu);font-size:13px}

@keyframes svpl{50%{opacity:.3}}
@media(max-width:900px){.sv-pipe{grid-template-columns:1fr 1fr}.sv-band{grid-template-columns:1fr 1fr}.sv-cl{grid-template-columns:1fr}}
@media(max-width:560px){.sv-pipe,.sv-band{grid-template-columns:1fr}.sv-rw{grid-template-columns:12px 1fr 44px}.sv-rw u{display:none}}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
"""

FORCE_LIGHT_JS = "() => { document.body.classList.remove('dark'); }"

# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
with gr.Blocks(title="SentinelVision Demo", theme=THEME, css=CSS, js=FORCE_LIGHT_JS) as demo:
    chain = gr.State([])
    gr.HTML(HERO)

    with gr.Row(equal_height=False):
        # ---------------- policy sidebar ----------------
        with gr.Column(scale=3, min_width=280, elem_classes="sv-card"):
            gr.HTML('<div class="sv-side-h">Policy</div>')
            required = gr.CheckboxGroup(
                choices=[(label, key) for key, label in PPE_ITEMS.items()],
                value=DEFAULT_REQUIRED,
                label="Required PPE",
                info="A missing required item makes the worker a violation.",
            )
            with gr.Row():
                camera = gr.Textbox(label="Camera", value="cam-01", max_lines=1)
                zone = gr.Textbox(label="Zone", value="gate-b", max_lines=1)
            gr.HTML('<div class="sv-side-h" style="margin-top:10px">Detector</div>')
            floor = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Detector floor",
                              info="Lowest confidence the model may report.")
            strict = gr.Checkbox(value=True, label="Apply per-class bars",
                                 info="Violations need 0.65, positives 0.50, persons 0.35.")

        # ---------------- main tabs ----------------
        with gr.Column(scale=9):
            with gr.Tabs():
                with gr.Tab("Image"):
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1, min_width=320, elem_classes="sv-card"):
                            inp = gr.Image(label="Input image", type="numpy", height=460)
                            run_btn = gr.Button("Check this image", variant="primary", elem_classes="sv-go")
                        with gr.Column(scale=1, min_width=320, elem_classes="sv-card"):
                            out_img = gr.Image(label="Annotated", type="numpy", interactive=False, height=460)
                    out_html = gr.HTML(EMPTY_HTML)
                    with gr.Accordion("Raw detections (JSON)", open=False):
                        out_json = gr.Code(label="Detections", language="json", interactive=False)

                with gr.Tab("Video · temporal vote"):
                    with gr.Row():
                        with gr.Column(elem_classes="sv-card"):
                            vid = gr.Video(label="Footage", sources=["upload"])
                            with gr.Row():
                                stride = gr.Slider(1, 30, value=5, step=1, label="Analyse every Nth frame")
                                max_frames = gr.Slider(10, 600, value=120, step=10, label="Max frames")
                            with gr.Row():
                                window = gr.Slider(3, 20, value=WINDOW_DEFAULT, step=1, label="Vote window")
                                quorum = gr.Slider(1, 20, value=QUORUM_DEFAULT, step=1, label="Quorum")
                            vid_btn = gr.Button("Run over footage", variant="primary", elem_classes="sv-go")
                        with gr.Column(elem_classes="sv-card"):
                            gallery = gr.Gallery(label="Frames where a violation was committed",
                                                 columns=2, height=360, object_fit="contain")
                    vid_html = gr.HTML(EMPTY_HTML.replace("Drop in a site photo.", "Drop in a short clip."))

                with gr.Tab("Audit trail"):
                    audit_badge = gr.HTML(chain_badge([]))
                    audit_text = gr.Textbox(
                        label="events.jsonl (editable, so you can test tamper detection)",
                        lines=14, max_lines=24, interactive=True, elem_id="sv-audit",
                        placeholder="Committed violations appear here, one hashed record per line.",
                    )
                    with gr.Row():
                        verify_btn = gr.Button("Verify chain", variant="primary", elem_classes="sv-go")
                        clear_btn = gr.Button("Clear log", elem_classes="sv-ghost")
                    audit_file = gr.File(label="Download events.jsonl", interactive=False)
                    gr.HTML(
                        '<p class="sv-foot">Each record carries a SHA-256 digest over its own contents and the '
                        "previous record's digest. Change any character above and press Verify to see the chain break.</p>"
                    )

                with gr.Tab("Policy & classes"):
                    gr.HTML(_taxonomy_html())

    gr.HTML(FOOTER)

    # ---------------- wiring ----------------
    audit_outs = [audit_text, audit_file, audit_badge]
    shared = [required, camera, zone, floor, strict]

    img_in = [inp, *shared, chain]
    img_out = [out_img, out_html, out_json, chain, *audit_outs]
    run_btn.click(run_image, inputs=img_in, outputs=img_out)
    inp.change(run_image, inputs=img_in, outputs=img_out)  # also run on upload / clear

    vid_btn.click(
        run_video,
        inputs=[vid, *shared, stride, max_frames, window, quorum, chain],
        outputs=[gallery, vid_html, chain, *audit_outs],
    )

    verify_btn.click(verify_text, inputs=audit_text, outputs=audit_badge)
    clear_btn.click(clear_chain, outputs=[chain, *audit_outs])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)