# =====================================================================
# RESOURCE MONITOR & CLOUD CONTAINER TELEMETRY (CPU / RAM / MEMORY)
# Real-Time diagnostics for Streamlit Community Cloud & Local environments
# =====================================================================
import os
import sys
import datetime
import streamlit as st

try:
    import psutil
except ImportError:
    psutil = None


def get_system_telemetry():
    """
    Retrieves real-time CPU, RAM, and process telemetry with zero CPU overhead.
    Works on Linux (Streamlit Cloud containers), macOS, and Windows.
    """
    telemetry = {
        "app_ram_mb": 0.0,
        "container_total_ram_mb": 1024.0,   # Standard Streamlit Cloud free tier quota
        "container_used_ram_mb": 0.0,
        "container_ram_pct": 0.0,
        "cpu_pct": 0.0,
        "cpu_count": os.cpu_count() or 1,
        "platform": "Linux (Streamlit Cloud)" if sys.platform.startswith("linux") else sys.platform,
        "status": "🟢 Optimal (Safe from Throttling)",
        "status_color": "#16a34a",
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
    }

    if psutil is not None:
        try:
            proc = psutil.Process()
            mem = proc.memory_info()
            telemetry["app_ram_mb"] = round(mem.rss / (1024 * 1024), 1)

            vm = psutil.virtual_memory()
            telemetry["container_total_ram_mb"] = round(vm.total / (1024 * 1024), 1)
            telemetry["container_used_ram_mb"] = round(vm.used / (1024 * 1024), 1)
            telemetry["container_ram_pct"] = round(vm.percent, 1)

            telemetry["cpu_pct"] = round(psutil.cpu_percent(interval=None), 1)
        except Exception:
            pass
    else:
        # Fallback reading directly from /proc on Linux Streamlit Cloud
        try:
            if os.path.exists("/proc/self/status"):
                with open("/proc/self/status", "r") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            telemetry["app_ram_mb"] = round(float(line.split()[1]) / 1024.0, 1)
                            break
            if os.path.exists("/proc/meminfo"):
                total_kb, avail_kb = 0, 0
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            total_kb = float(line.split()[1])
                        elif line.startswith("MemAvailable:"):
                            avail_kb = float(line.split()[1])
                if total_kb > 0:
                    telemetry["container_total_ram_mb"] = round(total_kb / 1024.0, 1)
                    used_kb = total_kb - avail_kb
                    telemetry["container_used_ram_mb"] = round(used_kb / 1024.0, 1)
                    telemetry["container_ram_pct"] = round((used_kb / total_kb) * 100, 1)
            if os.path.exists("/proc/loadavg"):
                with open("/proc/loadavg", "r") as f:
                    load_1m = float(f.read().split()[0])
                    telemetry["cpu_pct"] = round(min(100.0, (load_1m / max(1, telemetry["cpu_count"])) * 100), 1)
        except Exception:
            pass

    # Status classification against Streamlit Cloud quota limits (1024 MB RAM, 1 vCPU)
    app_ram = telemetry["app_ram_mb"]
    cpu = telemetry["cpu_pct"]

    if app_ram > 800 or cpu > 80:
        telemetry["status"] = "🔴 High Usage (Throttling Risk)"
        telemetry["status_color"] = "#dc2626"
    elif app_ram > 500 or cpu > 50:
        telemetry["status"] = "🟡 Moderate Usage"
        telemetry["status_color"] = "#d97706"
    else:
        telemetry["status"] = "🟢 Optimal (Safe from Throttling)"
        telemetry["status_color"] = "#16a34a"

    return telemetry


def render_resource_monitor_sidebar(key_suffix=""):
    """
    Renders an expandable real-time Cloud Resource Telemetry widget in the sidebar.
    """
    tel = get_system_telemetry()
    with st.sidebar.expander("⚡ Cloud Resource Monitor (CPU / RAM)", expanded=False):
        c1, c2 = st.columns(2)
        c1.metric("App Process RAM", f"{tel['app_ram_mb']:.1f} MB")
        c2.metric("CPU Load", f"{tel['cpu_pct']:.1f}%")

        # Visual progress bar towards 1024 MB Streamlit Cloud Limit
        pct_of_limit = min(100, int((tel['app_ram_mb'] / 1024.0) * 100))
        st.progress(pct_of_limit, text=f"Container Memory: {tel['app_ram_mb']:.0f} / 1024 MB ({pct_of_limit}%)")

        st.markdown(
            f"""
            <div style="background-color: {tel['status_color']}15; border: 1px solid {tel['status_color']}44; border-radius: 6px; padding: 6px 10px; margin-top: 6px; font-size: 0.78rem;">
                <b style="color: {tel['status_color']};">{tel['status']}</b><br>
                <span style="color: #64748b; font-size: 0.72rem;">Streamlit Cloud Quota: 1 vCPU • 1 GB RAM</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("🔄 Poll Resources", key=f"btn_poll_res_{key_suffix}", use_container_width=True):
            st.rerun()


def render_resource_monitor_card(key_suffix=""):
    """
    Renders an inline horizontal card showing real-time CPU & RAM health.
    """
    tel = get_system_telemetry()
    pct_of_limit = min(100, int((tel['app_ram_mb'] / 1024.0) * 100))
    st.markdown(
        f"""
        <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px; margin: 10px 0; display: flex; justify-content: space-between; align-items: center; font-size: 0.80rem;">
            <div>
                <b>💻 Cloud Resource Health:</b> App RAM: <b>{tel['app_ram_mb']:.1f} MB</b> / 1024 MB ({pct_of_limit}%) | CPU Load: <b>{tel['cpu_pct']:.1f}%</b>
            </div>
            <div>
                <span style="background-color: {tel['status_color']}20; color: {tel['status_color']}; font-weight: 700; padding: 3px 8px; border-radius: 4px; border: 1px solid {tel['status_color']}40;">
                    {tel['status']}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
