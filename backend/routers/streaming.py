"""Real-time streaming inference WebSocket router.

Supports two modes:
  1. **Demo mode** — server generates synthetic ECG/EEG waveforms and streams
     predictions back at the model's native sampling rate.
  2. **Device mode** — client pushes raw samples from a wearable device; server
     runs sliding-window inference and streams predictions back.

Protocol (JSON messages over WebSocket):
  Client → Server:
    { "type": "start", "model_id": "ecg_arrhythmia", "mode": "demo"|"device",
      "sampling_rate": 360, "heart_rate": 72 }
    { "type": "samples", "data": [0.1, 0.3, -0.2, ...] }
    { "type": "stop" }
    { "type": "configure_alerts", "classes": [1, 2], "threshold": 0.6 }

  Server → Client:
    { "type": "config", ... }          — session configuration summary
    { "type": "samples", "data": [...], "sr": 360 }  — signal chunk (demo)
    { "type": "prediction", ... }      — real-time prediction result
    { "type": "stats", ... }           — periodic session statistics
    { "type": "alert", ... }           — anomaly alert
    { "type": "error", "message": ... }
    { "type": "stopped" }
"""

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.config import MODEL_REGISTRY

router = APIRouter()


@router.websocket("/stream/ws")
async def streaming_websocket(ws: WebSocket):
    """Real-time streaming inference WebSocket endpoint."""
    await ws.accept()

    session = None
    demo_task = None
    running = False

    try:
        while True:
            msg = await ws.receive_json()
            msg_type = msg.get("type")

            # ── START ──────────────────────────────────────────
            if msg_type == "start":
                model_id = msg.get("model_id", "ecg_arrhythmia")
                mode = msg.get("mode", "demo")
                sr = msg.get("sampling_rate")

                if model_id not in MODEL_REGISTRY:
                    await ws.send_json({
                        "type": "error",
                        "message": f"Unknown model: {model_id}. Available: {list(MODEL_REGISTRY.keys())}",
                    })
                    continue

                model_info = MODEL_REGISTRY[model_id]

                from backend.services.streaming import StreamingSession
                session = StreamingSession(
                    model_id=model_id,
                    sampling_rate=sr or model_info["sampling_rate"],
                )

                # Default alerts: non-Normal classes for ECG
                if model_info["signal_type"] == "ecg":
                    session.configure_alerts([1, 2, 3, 4], threshold=0.6)
                elif model_info["signal_type"] == "eeg":
                    session.configure_alerts([], threshold=0.8)

                running = True

                await ws.send_json({
                    "type": "config",
                    "model_id": model_id,
                    "model_description": model_info["description"],
                    "signal_type": model_info["signal_type"],
                    "classes": model_info["classes"],
                    "sampling_rate": session.sampling_rate,
                    "segment_length": session.segment_length,
                    "stride": session.stride,
                    "mode": mode,
                })

                # Launch demo generator if demo mode
                if mode == "demo":
                    heart_rate = msg.get("heart_rate", 72)
                    demo_task = asyncio.create_task(
                        _demo_loop(ws, session, heart_rate)
                    )

            # ── SAMPLES (device mode) ──────────────────────────
            elif msg_type == "samples":
                if session is None:
                    await ws.send_json({"type": "error", "message": "Session not started. Send 'start' first."})
                    continue
                data = msg.get("data", [])
                if data:
                    session.push_samples(data)
                    # Send back any new predictions
                    for pred in session.get_new_predictions():
                        if pred.get("is_alert"):
                            await ws.send_json({**pred, "type": "alert"})
                        await ws.send_json(pred)

            # ── CONFIGURE ALERTS ───────────────────────────────
            elif msg_type == "configure_alerts":
                if session:
                    session.configure_alerts(
                        msg.get("classes", []),
                        msg.get("threshold", 0.5),
                    )
                    await ws.send_json({"type": "info", "message": "Alerts configured."})

            # ── STOP ───────────────────────────────────────────
            elif msg_type == "stop":
                running = False
                if demo_task and not demo_task.done():
                    demo_task.cancel()
                    try:
                        await demo_task
                    except asyncio.CancelledError:
                        pass
                    demo_task = None

                stats = session.get_stats() if session else {}
                await ws.send_json({"type": "stopped", **stats})
                session = None

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        running = False
        if demo_task and not demo_task.done():
            demo_task.cancel()


async def _demo_loop(ws: WebSocket, session, heart_rate: float = 72):
    """Generate and stream synthetic signal data at real-time pace."""
    from backend.services.streaming import generate_ecg_samples, generate_eeg_samples

    sr = session.sampling_rate
    sig_type = session.signal_type

    # Send chunks at ~20 Hz (50ms intervals)
    chunk_size = max(1, int(sr / 20))
    interval = chunk_size / sr  # seconds per chunk

    stats_counter = 0
    STATS_EVERY = 40  # send stats every ~2 seconds

    try:
        while True:
            # Generate chunk
            if sig_type == "ecg":
                chunk = generate_ecg_samples(chunk_size, sr, heart_rate=heart_rate)
            elif sig_type == "eeg":
                chunk = generate_eeg_samples(chunk_size, sr)
            else:
                chunk = generate_ecg_samples(chunk_size, sr, heart_rate=heart_rate)

            # Send raw samples to client for visualization
            await ws.send_json({
                "type": "samples",
                "data": chunk.tolist(),
                "sr": sr,
            })

            # Feed into inference engine
            session.push_samples(chunk)

            # Send any new predictions
            for pred in session.get_new_predictions():
                if pred.get("is_alert"):
                    await ws.send_json({**pred, "type": "alert"})
                await ws.send_json(pred)

            # Periodic stats
            stats_counter += 1
            if stats_counter >= STATS_EVERY:
                stats_counter = 0
                await ws.send_json({"type": "stats", **session.get_stats()})

            await asyncio.sleep(interval)

    except asyncio.CancelledError:
        return
    except WebSocketDisconnect:
        return
