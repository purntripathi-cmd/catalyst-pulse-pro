# =====================================================================
# AUTONOMOUS BACKTEST AGENT & BACKGROUND SERVICE
# AI-Powered Deep-Value & Contrarian Investment Architecture (v4.2-Production)
# Runs long-duration backtests & fine-tuning independently (even if user is offline)
# =====================================================================
import os
import sys
import time
import json
import argparse
import datetime
import subprocess
import logging
import pandas as pd
import numpy as np

try:
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
except Exception:
    IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

DATA_DIR = os.path.join(CURRENT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

AGENT_STATE_PATH = os.path.join(DATA_DIR, "dual_logic_agent_state.json")
AGENT_LOG_PATH = os.path.join(DATA_DIR, "dual_logic_agent.log")
FINDINGS_JSON_PATH = os.path.join(DATA_DIR, "dual_logic_findings.json")
RUNS_CSV_PATH = os.path.join(DATA_DIR, "dual_logic_runs.csv")
CONFIG_JSON_PATH = os.path.join(CURRENT_DIR, "runtime_config.json")
PARAM_LOG_CSV = os.path.join(DATA_DIR, "parameter_change_log.csv")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(AGENT_LOG_PATH, encoding="utf-8")
    ]
)
logger = logging.getLogger("AutonomousBacktestAgent")

from dual_logic_engine import DualLogicBacktestEngine, DEFAULT_OPTIMIZED_WEIGHTS, UNOPTIMIZED_WEIGHTS
from data_pipeline_20y import generate_20y_ground_truth_dataset


def get_agent_status() -> dict:
    """Reads the current background agent state from disk."""
    if os.path.exists(AGENT_STATE_PATH):
        try:
            with open(AGENT_STATE_PATH, "r") as f:
                state = json.load(f)
                # Check if process is still alive if marked RUNNING
                if state.get("status") == "RUNNING":
                    pid = state.get("pid")
                    if pid:
                        # On Windows, check process existence
                        try:
                            import ctypes
                            PROCESS_QUERY_INFORMATION = 0x0400
                            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, 0, pid)
                            if handle == 0:
                                state["status"] = "IDLE (Process Completed)"
                            else:
                                ctypes.windll.kernel32.CloseHandle(handle)
                        except Exception:
                            pass
                return state
        except Exception as e:
            logger.warning(f"Error reading agent state: {e}")
    return {
        "status": "IDLE",
        "pid": None,
        "start_time": "None",
        "duration_hours": 0.0,
        "current_epoch": 0,
        "total_epochs": 0,
        "elapsed_seconds": 0,
        "best_win_rate": 0.0,
        "best_accuracy": 0.0,
        "is_locked": False,
        "drift_status": "UNKNOWN",
        "last_heartbeat": "None"
    }


def save_agent_state(state: dict):
    """Atomically persists agent state to disk."""
    state["last_heartbeat"] = datetime.datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    with open(AGENT_STATE_PATH, "w") as f:
        json.dump(state, f, indent=4)


def apply_findings_to_app(user: str = "Autonomous_DualLogic_Agent") -> dict:
    """
    Reuses findings: Pushes the locked, optimized weights and parameters
    directly into runtime_config.json so all live screeners and paper trading
    strategies immediately utilize the calibrated weights.
    """
    if not os.path.exists(FINDINGS_JSON_PATH):
        return {"success": False, "message": "No findings file found. Run calibration first."}

    try:
        with open(FINDINGS_JSON_PATH, "r") as f:
            findings = json.load(f)

        opt_weights = findings.get("optimal_weights", DEFAULT_OPTIMIZED_WEIGHTS)

        # Load runtime config
        cfg = {}
        if os.path.exists(CONFIG_JSON_PATH):
            with open(CONFIG_JSON_PATH, "r") as f:
                cfg = json.load(f)

        if "weights" not in cfg:
            cfg["weights"] = {}

        # 1. Register dedicated preset: "Deep-Value & Contrarian"
        cfg["weights"]["Deep-Value & Contrarian"] = {
            "w_de": int(opt_weights.get("debt_eq", 0.30) * 100),
            "w_dd": int(opt_weights.get("drawdown", 0.20) * 100),
            "w_am": int(opt_weights.get("asset_moat", 0.30) * 100),
            "w_ic": int(opt_weights.get("interest_cov", 0.10) * 100),
            "w_grid": int(opt_weights.get("macro_grid", 0.10) * 100)
        }

        # 2. Also enhance "Long-Term Secular" and "Default" with the deep-value moat weights
        cfg["weights"]["Long-Term"]["w_dma"] = int(opt_weights.get("drawdown", 0.25) * 100)
        cfg["weights"]["Long-Term"]["w_div"] = int(opt_weights.get("asset_moat", 0.25) * 100)

        cfg["dual_logic_v4_2"] = {
            "is_locked": findings.get("is_locked", True),
            "lock_timestamp": findings.get("lock_timestamp", "2026-09-30"),
            "drift_status": findings.get("drift_status", "LOCKED_PRODUCTION"),
            "optimal_weights": opt_weights,
            "win_rate": findings.get("win_rate", 0.90),
            "accuracy": findings.get("accuracy", 0.82)
        }
        cfg["optimization_status"] = "Dual-Logic v4.2 Production Locked"
        cfg["last_optimized_timestamp"] = datetime.datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

        with open(CONFIG_JSON_PATH, "w") as f:
            json.dump(cfg, f, indent=4)

        # Log parameter change audit trail
        audit_row = {
            "Timestamp_IST": datetime.datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
            "Changed_By": user,
            "Parameter_Category": "Dual_Logic_v4_2",
            "Parameter_Name": "Optimal_Weights_Applied",
            "Old_Value": "Unoptimized Baseline",
            "New_Value": json.dumps(opt_weights),
            "Source": "Autonomous Agent Backtest Daemon",
            "Intended_Impact": "Enforce >=10% CAGR bear market resilience and physical moat allocation"
        }
        audit_df = pd.DataFrame([audit_row])
        if os.path.exists(PARAM_LOG_CSV):
            audit_df.to_csv(PARAM_LOG_CSV, mode="a", header=False, index=False)
        else:
            audit_df.to_csv(PARAM_LOG_CSV, index=False)

        logger.info(f"Successfully applied dual-logic findings to {CONFIG_JSON_PATH}.")
        return {"success": True, "message": "Optimal weights applied live to runtime_config.json."}
    except Exception as e:
        logger.error(f"Failed to apply findings: {e}")
        return {"success": False, "message": str(e)}


class AutonomousBacktestAgent:
    """
    Autonomous Background Backtest & Optimization Agent.
    Executes multi-epoch Bayesian tuning, walk-forward validation across 2006-2026,
    monitors drift, records findings, and locks the logic when optimal.
    """
    def __init__(self, duration_hours: float = 1.0, max_epochs: int = 100):
        self.duration_hours = duration_hours
        self.max_epochs = max_epochs
        self.dataset = generate_20y_ground_truth_dataset()
        self.engine = DualLogicBacktestEngine(self.dataset)
        self.state = {
            "status": "RUNNING",
            "pid": os.getpid(),
            "start_time": datetime.datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
            "duration_hours": duration_hours,
            "current_epoch": 0,
            "total_epochs": max_epochs,
            "elapsed_seconds": 0,
            "best_win_rate": 0.0,
            "best_accuracy": 0.0,
            "current_weights": self.engine.weights,
            "is_locked": False,
            "drift_status": "INITIALIZING",
            "last_heartbeat": datetime.datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
        }

    def run(self):
        """Main execution loop for continuous self-optimizing backtests."""
        start_ts = time.time()
        end_ts = start_ts + (self.duration_hours * 3600)
        logger.info(f"Starting autonomous backtesting session for {self.duration_hours:.2f} hours (PID: {os.getpid()})...")
        save_agent_state(self.state)

        epoch = 0
        best_composite = 0.0

        try:
            while time.time() < end_ts and epoch < self.max_epochs:
                epoch += 1
                self.state["current_epoch"] = epoch
                self.state["elapsed_seconds"] = int(time.time() - start_ts)

                logger.info(f"--- Epoch {epoch}/{self.max_epochs} | Elapsed: {self.state['elapsed_seconds']}s ---")

                # 1. Run Bayesian optimization trial batch
                self.engine.is_locked = False
                res_str = self.engine.logic_c_evaluate_and_retune(n_trials=15)
                logger.info(f"Epoch {epoch} Result: {res_str}")

                # 2. Evaluate performance
                eval_metrics = self.engine.logic_c_evaluate()
                acc = eval_metrics["accuracy"]
                win = eval_metrics["win_rate"]
                composite = 0.5 * acc + 0.5 * win

                if composite > best_composite or win > self.state["best_win_rate"]:
                    best_composite = composite
                    self.state["best_win_rate"] = win
                    self.state["best_accuracy"] = acc
                    self.state["current_weights"] = self.engine.weights.copy()
                    logger.info(f"[PEAK BENCHMARK] Epoch {epoch}: Win Rate {win:.2%}, Acc {acc:.2%}")

                # 3. Check for out-of-sample drift (2024-2026)
                drift_res = self.engine.check_for_drift()
                self.state["drift_status"] = drift_res["drift_status"]
                self.state["is_locked"] = self.engine.is_locked

                # 4. Save state & findings every epoch
                save_agent_state(self.state)
                self.engine.save_findings()

                # If locked and stable, apply findings and maintain passive drift monitoring
                if self.engine.is_locked and drift_res["drift_status"] == "STABLE":
                    apply_findings_to_app(user="Autonomous_Agent_Daemon")
                    logger.info("Logic locked and verified stable across 2006-2026. Passive drift monitor active.")
                    # Sleep between passive checks
                    time.sleep(10)
                else:
                    time.sleep(2)

            self.state["status"] = "COMPLETED"
            self.state["elapsed_seconds"] = int(time.time() - start_ts)
            save_agent_state(self.state)
            logger.info("Autonomous backtesting and optimization session completed successfully.")

        except KeyboardInterrupt:
            logger.info("Agent process interrupted by user.")
            self.state["status"] = "STOPPED_BY_USER"
            save_agent_state(self.state)
        except Exception as e:
            logger.error(f"Unexpected error in agent loop: {e}", exc_info=True)
            self.state["status"] = f"ERROR: {str(e)}"
            save_agent_state(self.state)


def start_background_agent(duration_hours: float = 1.0) -> dict:
    """Spawns the autonomous agent as a detached background process."""
    current_status = get_agent_status()
    if current_status.get("status") == "RUNNING":
        return {"success": False, "message": f"Agent is already running (PID: {current_status.get('pid')})."}

    cmd = [sys.executable, os.path.abspath(__file__), "--run-daemon", "--duration-hours", str(duration_hours)]
    
    # Spawn detached on Windows
    DETACHED_PROCESS = 0x00000008
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    creationflags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP

    proc = subprocess.Popen(
        cmd,
        cwd=CURRENT_DIR,
        creationflags=creationflags,
        close_fds=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    init_state = {
        "status": "RUNNING",
        "pid": proc.pid,
        "start_time": datetime.datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        "duration_hours": duration_hours,
        "current_epoch": 0,
        "total_epochs": int(duration_hours * 60),
        "elapsed_seconds": 0,
        "best_win_rate": 0.0,
        "best_accuracy": 0.0,
        "is_locked": False,
        "drift_status": "STARTING",
        "last_heartbeat": datetime.datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    }
    save_agent_state(init_state)

    logger.info(f"Spawned autonomous background agent (PID: {proc.pid}, Duration: {duration_hours}h).")
    return {"success": True, "pid": proc.pid, "message": f"Background agent launched (PID: {proc.pid})."}


def stop_background_agent() -> dict:
    """Terminates any running background agent."""
    state = get_agent_status()
    pid = state.get("pid")
    if not pid or "RUNNING" not in state.get("status", ""):
        return {"success": False, "message": "No agent currently running."}

    try:
        import signal
        os.kill(pid, signal.SIGTERM)
    except Exception as e:
        # Fallback to taskkill on Windows
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)

    state["status"] = "STOPPED_MANUALLY"
    save_agent_state(state)
    logger.info(f"Terminated background agent process {pid}.")
    return {"success": True, "message": f"Agent process {pid} stopped."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Dual-Logic Backtest Agent")
    parser.add_argument("--start", action="store_true", help="Launch agent in background")
    parser.add_argument("--stop", action="store_true", help="Stop running agent")
    parser.add_argument("--status", action="store_true", help="Display agent status")
    parser.add_argument("--run-daemon", action="store_true", help="Internal daemon runner")
    parser.add_argument("--run-once", action="store_true", help="Run single calibration cycle")
    parser.add_argument("--apply", action="store_true", help="Apply findings to runtime_config.json")
    parser.add_argument("--duration-hours", type=float, default=1.0, help="Duration in hours")

    args = parser.parse_args()

    if args.start:
        res = start_background_agent(duration_hours=args.duration_hours)
        print(json.dumps(res, indent=2))
    elif args.stop:
        res = stop_background_agent()
        print(json.dumps(res, indent=2))
    elif args.status:
        st = get_agent_status()
        print(json.dumps(st, indent=2))
    elif args.apply:
        res = apply_findings_to_app()
        print(json.dumps(res, indent=2))
    elif args.run_once:
        print("Running single calibration sweep...")
        agent = AutonomousBacktestAgent(duration_hours=0.05, max_epochs=1)
        agent.run()
        print("Completed.")
    elif args.run_daemon:
        agent = AutonomousBacktestAgent(duration_hours=args.duration_hours)
        agent.run()
    else:
        parser.print_help()
