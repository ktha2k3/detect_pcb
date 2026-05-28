import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "app.db"
USERS_JSON = APP_DIR / "users.json"
DEFAULT_DEFECT_TYPES = [
    "Dry_joint",
    "Incorrect_installation",
    "PCB_damage",
    "Short_circuit",
]


def get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """
    )

    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS worker_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        login_at TEXT NOT NULL,
        logout_at TEXT,
        FOREIGN KEY(username) REFERENCES users(username)
    )
    """
    )

    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        started_at TEXT NOT NULL,
        completed_at TEXT,
        image_count INTEGER DEFAULT 0,
        detection_count INTEGER DEFAULT 0
    )
    """
    )

    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL,
        image_filename TEXT NOT NULL,
        product_name TEXT,
        class_name TEXT,
        confidence REAL,
        bbox TEXT,
        original_image_path TEXT,
        annotated_image_path TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(run_id) REFERENCES runs(id)
    )
    """
    )

    ensure_detection_columns(cur)

    conn.commit()
    conn.close()
    # migrate existing users.json if present
    if USERS_JSON.exists():
        migrate_users_json()


def migrate_users_json():
    try:
        with open(USERS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return

    conn = get_conn()
    cur = conn.cursor()
    for username, info in data.items():
        cur.execute(
            "INSERT OR IGNORE INTO users(username, password_hash, role, created_at) VALUES(?,?,?,?)",
            (username, info.get("password_hash"), info.get("role", "worker"), info.get("created_at", datetime.now().isoformat())),
        )
    conn.commit()
    conn.close()


def ensure_detection_columns(cur) -> None:
    cur.execute("PRAGMA table_info(detections)")
    existing_columns = {row[1] for row in cur.fetchall()}
    if "original_image_path" not in existing_columns:
        cur.execute("ALTER TABLE detections ADD COLUMN original_image_path TEXT")
    if "annotated_image_path" not in existing_columns:
        cur.execute("ALTER TABLE detections ADD COLUMN annotated_image_path TEXT")


def add_user(username: str, password_hash: str, role: str, created_at: Optional[str] = None):
    created_at = created_at or datetime.now().isoformat()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO users(username, password_hash, role, created_at) VALUES(?,?,?,?)",
        (username, password_hash, role, created_at),
    )
    conn.commit()
    conn.close()


def delete_user(username: str) -> bool:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM users WHERE username = ?", (username,))
    changed = cur.rowcount
    conn.commit()
    conn.close()
    # also remove from users.json if present
    if USERS_JSON.exists():
        try:
            data = json.loads(USERS_JSON.read_text(encoding="utf-8"))
            if username in data:
                del data[username]
                USERS_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
    return bool(changed)


def get_user(username: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT username, role, created_at FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return dict(row)


def record_session_login(username: str) -> int:
    conn = get_conn()
    cur = conn.cursor()
    now = datetime.now().isoformat()
    cur.execute("INSERT INTO worker_sessions(username, login_at) VALUES(?,?)", (username, now))
    session_id = cur.lastrowid
    conn.commit()
    conn.close()
    return session_id


def record_session_logout(session_id: int) -> None:
    conn = get_conn()
    cur = conn.cursor()
    now = datetime.now().isoformat()
    cur.execute("UPDATE worker_sessions SET logout_at = ? WHERE id = ?", (now, session_id))
    conn.commit()
    conn.close()


def create_run(username: str, started_at: Optional[str] = None) -> int:
    conn = get_conn()
    cur = conn.cursor()
    started_at = started_at or datetime.now().isoformat()
    cur.execute("INSERT INTO runs(username, started_at) VALUES(?,?)", (username, started_at))
    run_id = cur.lastrowid
    conn.commit()
    conn.close()
    return run_id


def complete_run(run_id: int, image_count: int, detection_count: int) -> None:
    conn = get_conn()
    cur = conn.cursor()
    now = datetime.now().isoformat()
    cur.execute(
        "UPDATE runs SET completed_at = ?, image_count = ?, detection_count = ? WHERE id = ?",
        (now, image_count, detection_count, run_id),
    )
    conn.commit()
    conn.close()


def add_detection(
    run_id: int,
    image_filename: str,
    product_name: str,
    class_name: str,
    confidence: float,
    bbox: str,
    original_image_path: Optional[str] = None,
    annotated_image_path: Optional[str] = None,
) -> None:
    conn = get_conn()
    cur = conn.cursor()
    now = datetime.now().isoformat()
    cur.execute(
        "INSERT INTO detections(run_id, image_filename, product_name, class_name, confidence, bbox, original_image_path, annotated_image_path, created_at) VALUES(?,?,?,?,?,?,?,?,?)",
        (run_id, image_filename, product_name, class_name, confidence, bbox, original_image_path, annotated_image_path, now),
    )
    conn.commit()
    conn.close()


def search_detections(
    class_name: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    product_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    q = "SELECT d.id, d.run_id, d.image_filename, d.product_name, d.class_name, d.confidence, d.bbox, d.original_image_path, d.annotated_image_path, d.created_at, r.username FROM detections d JOIN runs r ON d.run_id = r.id WHERE 1=1"
    params: List[Any] = []
    if class_names:
        placeholders = ",".join(["?"] * len(class_names))
        q += f" AND d.class_name IN ({placeholders})"
        params.extend(class_names)
    if class_name:
        q += " AND d.class_name LIKE ?"
        params.append(f"%{class_name}%")
    if product_name:
        q += " AND d.product_name LIKE ?"
        params.append(f"%{product_name}%")
    q += " ORDER BY d.created_at DESC"
    cur.execute(q, params)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def list_defect_types() -> List[str]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT class_name FROM detections WHERE class_name IS NOT NULL AND TRIM(class_name) <> '' ORDER BY class_name"
    )
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    defect_types = list(DEFAULT_DEFECT_TYPES)
    for label in rows:
        if label not in defect_types:
            defect_types.append(label)
    return defect_types


def list_workers() -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT username, role, created_at FROM users WHERE role = 'worker' ORDER BY username")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_worker_sessions(username: str) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, login_at, logout_at FROM worker_sessions WHERE username = ? ORDER BY login_at DESC",
        (username,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_worker_products(username: str) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT d.product_name, COUNT(*) as cnt FROM detections d JOIN runs r ON d.run_id = r.id WHERE r.username = ? GROUP BY d.product_name ORDER BY cnt DESC",
        (username,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_worker_detections(
    username: str,
    class_name: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    product_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    q = "SELECT d.id, d.run_id, d.image_filename, d.product_name, d.class_name, d.confidence, d.bbox, d.original_image_path, d.annotated_image_path, d.created_at FROM detections d JOIN runs r ON d.run_id = r.id WHERE r.username = ?"
    params: List[Any] = [username]
    if class_names:
        placeholders = ",".join(["?"] * len(class_names))
        q += f" AND d.class_name IN ({placeholders})"
        params.extend(class_names)
    if class_name:
        q += " AND d.class_name LIKE ?"
        params.append(f"%{class_name}%")
    if product_name:
        q += " AND d.product_name LIKE ?"
        params.append(f"%{product_name}%")
    q += " ORDER BY d.created_at DESC"
    cur.execute(q, params)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows
