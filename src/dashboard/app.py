"""
ProxiSense Streamlit Dashboard
Real-time visualization of detection, tracking, intent prediction, and alerts.

Usage:
    streamlit run src/dashboard/app.py
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import altair as alt
from pathlib import Path
from collections import deque
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pipeline import ProxiSensePipeline


st.set_page_config(
    page_title="ProxiSense - ADAS Intent Prediction",
    page_icon="🚗",
    layout="wide"
)


def init_session_state():
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "running" not in st.session_state:
        st.session_state.running = False
    if "frame_placeholder" not in st.session_state:
        st.session_state.frame_placeholder = None
    if "metrics_history" not in st.session_state:
        st.session_state.metrics_history = deque(maxlen=100)
    if "alert_log" not in st.session_state:
        st.session_state.alert_log = deque(maxlen=50)
    if "fps_history" not in st.session_state:
        st.session_state.fps_history = deque(maxlen=100)


def render_sidebar():
    st.sidebar.title("ProxiSense")
    st.sidebar.caption("Real-Time Intent Prediction for ADAS")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Controls")

    source_type = st.sidebar.selectbox(
        "Input Source",
        ["Webcam", "Video File", "CARLA Simulator"]
    )

    source_path = None
    if source_type == "Video File":
        source_path = st.sidebar.text_input(
            "Video Path",
            "data/sample_videos/test.mp4"
        )

    use_webcam = source_type == "Webcam"
    use_carla = source_type == "CARLA Simulator"

    st.sidebar.markdown("---")
    st.sidebar.subheader("Settings")

    conf_threshold = st.sidebar.slider("Detection Confidence", 0.1, 0.9, 0.5, 0.05)
    intent_threshold = st.sidebar.slider("Alert Confidence", 0.5, 0.95, 0.75, 0.05)
    show_trajectories = st.sidebar.checkbox("Show Trajectories", True)
    show_zone_overlay = st.sidebar.checkbox("Show TTC Zones", True)

    col1, col2 = st.sidebar.columns(2)
    start_btn = col1.button("▶ Start", use_container_width=True)
    stop_btn = col2.button("■ Stop", use_container_width=True)

    return {
        "use_webcam": use_webcam,
        "source_path": source_path,
        "use_carla": use_carla,
        "conf_threshold": conf_threshold,
        "intent_threshold": intent_threshold,
        "show_trajectories": show_trajectories,
        "show_zone_overlay": show_zone_overlay,
        "start_btn": start_btn,
        "stop_btn": stop_btn
    }


def render_metrics(alerts, fps, inference_time):
    col1, col2, col3, col4 = st.columns(4)

    max_alert = "SAFE"
    for alert in alerts:
        if alert.zone == "RED":
            max_alert = "RED"
        elif alert.zone == "AMBER" and max_alert != "RED":
            max_alert = "AMBER"
        elif alert.zone == "GREEN" and max_alert not in ("RED", "AMBER"):
            max_alert = "GREEN"

    alert_color = {"SAFE": "green", "GREEN": "green", "AMBER": "orange", "RED": "red"}
    col1.metric("System Status", max_alert, delta=None)
    col1.markdown(
        f"<div style='background:{alert_color[max_alert]};height:4px;border-radius:2px;'></div>",
        unsafe_allow_html=True
    )

    col2.metric("FPS", f"{fps:.1f}")
    col3.metric("Inference", f"{inference_time:.1f} ms")
    col4.metric("Agents Tracked", len(alerts))

    if alerts:
        st.markdown("### Active Alerts")
        alert_data = []
        for a in alerts:
            alert_data.append({
                "ID": a.track_id,
                "Class": a.class_name,
                "Intent": a.intent_label,
                "Confidence": f"{a.intent_confidence:.2f}",
                "TTC (s)": f"{a.ttc:.1f}" if a.ttc != float('inf') else "∞",
                "Zone": a.zone,
                "Brake": "⚠️" if a.braking_signal else "—"
            })
        st.dataframe(pd.DataFrame(alert_data), use_container_width=True, hide_index=True)


def render_plots(metrics_history):
    if len(metrics_history) < 2:
        st.info("Waiting for data...")
        return

    df = pd.DataFrame(metrics_history)

    col1, col2 = st.columns(2)

    with col1:
        fps_chart = alt.Chart(df).mark_line(color="blue").encode(
            x=alt.X("frame:Q", title="Frame"),
            y=alt.Y("fps:Q", title="FPS", scale=alt.Scale(zero=False))
        ).properties(height=200, title="Inference Performance")
        st.altair_chart(fps_chart, use_container_width=True)

    with col2:
        if "agents" in df.columns:
            agents_chart = alt.Chart(df).mark_area(color="green", opacity=0.3).encode(
                x=alt.X("frame:Q", title="Frame"),
                y=alt.Y("agents:Q", title="Agents Tracked")
            ).properties(height=200, title="Objects Tracked")
            st.altair_chart(agents_chart, use_container_width=True)


def main():
    init_session_state()
    controls = render_sidebar()

    main_col, right_col = st.columns([3, 1])

    with main_col:
        st.subheader("Live Feed")
        frame_placeholder = st.empty()

    with right_col:
        st.subheader("Metrics")
        metrics_placeholder = st.empty()

    plot_placeholder = st.empty()
    alert_placeholder = st.empty()

    if controls["start_btn"] or st.session_state.running:
        if controls["stop_btn"]:
            st.session_state.running = False
            if st.session_state.pipeline:
                st.session_state.pipeline.release()
                st.session_state.pipeline = None
            st.rerun()

        if st.session_state.pipeline is None:
            cfg_path = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
            st.session_state.pipeline = ProxiSensePipeline(
                config_path=str(cfg_path),
                use_webcam=controls["use_webcam"],
                video_path=controls["source_path"] if not controls["use_webcam"] else None
            )
            st.session_state.running = True

        pipeline = st.session_state.pipeline
        frame_count = 0

        while st.session_state.running:
            frame, alerts, metrics = pipeline.process_frame()

            if frame is None:
                st.warning("End of video or no frame received")
                break

            if controls["show_trajectories"]:
                frame = pipeline.render_trajectories(frame)

            if controls["show_zone_overlay"]:
                frame = pipeline.render_zone_overlay(frame)

            frame = pipeline.render_alerts(frame, alerts)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            st.session_state.metrics_history.append({
                "frame": frame_count,
                "fps": metrics.get("fps", 0),
                "agents": len(alerts),
                "inference_ms": metrics.get("inference_time", 0)
            })

            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            with metrics_placeholder.container():
                render_metrics(alerts, metrics.get("fps", 0), metrics.get("inference_time", 0))

            with plot_placeholder.container():
                render_plots(list(st.session_state.metrics_history))

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        pipeline.release()
        st.session_state.running = False

    else:
        st.info("Configure settings and click **Start** to begin the pipeline.")


if __name__ == "__main__":
    main()
