from __future__ import annotations

import hashlib
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from inference import load_model, predict_one
from db import (
    init_db,
    add_user,
    delete_user,
    get_user,
    record_session_login,
    record_session_logout,
    create_run,
    complete_run,
    add_detection,
    search_detections,
    list_defect_types,
    list_workers,
    get_worker_sessions,
    get_worker_products,
    get_worker_detections,
)


APP_DIR = Path(__file__).resolve().parent
AUTH_FILE = APP_DIR / "users.json"
RUNS_DIR = APP_DIR / "ui_runs"
DETECT_LABELS_VI = {
    "Dry_joint": "Mối hàn khô",
    "Incorrect_installation": "Lắp đặt sai",
    "PCB_damage": "Hư hỏng PCB",
    "Short_circuit": "Ngắn mạch",
}


def load_users() -> dict:
    if not AUTH_FILE.exists():
        return {}

    try:
        return json.loads(AUTH_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_users(users: dict) -> None:
    AUTH_FILE.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def register_user(username: str, password: str, role: str = "worker") -> tuple[bool, str]:
    username = username.strip()
    if not username or not password:
        return False, "Username and password are required."

    users = load_users()
    if username in users:
        return False, "Username already exists."
    users[username] = {
        "password_hash": hash_password(password),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "role": role,
    }
    save_users(users)
    return True, "Account created successfully."


def validate_login(username: str, password: str) -> bool:
    users = load_users()
    user = users.get(username.strip())
    if not user:
        return False
    return user.get("password_hash") == hash_password(password)


def file_to_array(uploaded_file) -> np.ndarray:
    image = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")
    return np.array(image)


def format_defect_label(label: str) -> str:
    return DETECT_LABELS_VI.get(label, label.replace("_", " ").strip().title())


def build_defect_label_map(defect_types: list[str]) -> dict[str, str]:
    return {format_defect_label(label): label for label in defect_types}


def translate_detection_rows(rows: list[dict]) -> list[dict]:
    translated_rows = []
    for row in rows:
        translated = dict(row)
        if translated.get("class_name"):
            translated["class_name"] = format_defect_label(str(translated["class_name"]))
        translated_rows.append(translated)
    return translated_rows


def translate_grouped_detection_rows(rows: list[dict]) -> list[dict]:
    translated_rows = []
    for row in rows:
        translated = dict(row)
        defect_summary = translated.get("defect_summary", "")
        if defect_summary and defect_summary != "-":
            translated_labels = [format_defect_label(part.strip()) for part in defect_summary.split(",") if part.strip()]
            translated["class_name"] = ", ".join(translated_labels)
        else:
            translated["class_name"] = "-"
        translated_rows.append(translated)
    return translated_rows


def group_detection_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[str, dict] = {}
    for row in rows:
        group_key = str(
            row.get("original_image_path")
            or row.get("annotated_image_path")
            or row.get("run_id")
            or row.get("image_filename")
        )
        group = grouped.setdefault(
            group_key,
            {
                "id": row.get("id"),
                "run_id": row.get("run_id"),
                "image_filename": row.get("image_filename"),
                "product_name": row.get("product_name"),
                "username": row.get("username"),
                "original_image_path": row.get("original_image_path"),
                "annotated_image_path": row.get("annotated_image_path"),
                "defects": [],
            },
        )
        group["defects"].append(
            {
                "class_name": row.get("class_name"),
                "confidence": row.get("confidence"),
                "bbox": row.get("bbox"),
            }
        )

    grouped_rows: list[dict] = []
    for group in grouped.values():
        defect_names = [str(defect.get("class_name", "")) for defect in group["defects"] if defect.get("class_name")]
        group["defect_count"] = len(group["defects"])
        group["defect_summary"] = ", ".join(sorted(set(defect_names))) if defect_names else "-"
        grouped_rows.append(group)

    return grouped_rows


def render_defect_list(detections: list[dict]) -> None:
    if not detections:
        st.info("Không phát hiện lỗi nào.")
        return

    for detection in detections:
        st.markdown(
            f"- **{detection.get('class_name', 'Lỗi')}**: độ tin cậy {float(detection.get('confidence', 0.0)):.2f}"
        )


def save_rgb_image(image_array: np.ndarray, path: Path) -> None:
    Image.fromarray(image_array).save(path)


def render_detection_previews(rows: list[dict], title: str, limit: int = 6, key_prefix: str = "preview") -> None:
    st.subheader(title)
    preview_rows = rows[:limit]
    if not preview_rows:
        st.info("No detections available for preview.")
        return

    labels = [
        f"{index + 1}. {row.get('image_filename', '')} | {row.get('product_name', '')} | {row.get('username', '')} | {row.get('defect_count', 0)} lỗi"
        for index, row in enumerate(preview_rows)
    ]
    selected_index = st.selectbox(
        "Chọn dòng để xem ảnh",
        list(range(len(preview_rows))),
        format_func=lambda index: labels[index],
        key=f"{key_prefix}_selector",
    )
    row = preview_rows[selected_index]

    with st.container(border=True):
        st.write(
            f"{row.get('image_filename', '')} | {row.get('product_name', '')} | {row.get('username', '')} | {row.get('defect_count', 0)} lỗi"
        )
        left, right = st.columns(2)
        original_path = row.get("original_image_path")
        annotated_path = row.get("annotated_image_path")

        if original_path and Path(original_path).exists():
            left.image(original_path, caption="Trước detect", use_container_width=True)
        else:
            left.info("Không có ảnh gốc đã lưu")

        if annotated_path and Path(annotated_path).exists():
            right.image(annotated_path, caption="Sau detect", use_container_width=True)
        else:
            right.info("Không có ảnh sau detect đã lưu")

        st.markdown("**Các lỗi phát hiện:**")
        render_defect_list(row.get("defects", []))


def render_detection_summary(item: dict) -> None:
    detections = item.get("detections", [])
    if not detections:
        st.info("Không phát hiện lỗi nào.")
        return

    st.markdown("**Các lỗi phát hiện:**")
    render_defect_list(detections)
    st.caption("Ảnh gốc và ảnh sau detect đã hiển thị phía trên.")


def init_state() -> None:
    st.session_state.setdefault("authenticated", False)
    st.session_state.setdefault("username", "")
    st.session_state.setdefault("role", "")
    st.session_state.setdefault("session_id", None)


def logout() -> None:
    st.session_state["authenticated"] = False
    st.session_state["username"] = ""
    st.session_state["role"] = ""


@st.cache_resource
def get_model():
    return load_model()


def render_auth_screen() -> None:
    # Using Streamlit default styling; custom CSS removed to restore original background

    st.title("PCB Batch Tester")
    st.caption("Create an account or sign in, then upload multiple images for batch inference.")

    # ensure DB initialized and migrate existing users.json
    init_db()
    users = load_users()
    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Login")
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Login")

        if submitted:
            if validate_login(username, password):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username.strip()
                # set role in session
                users_data = load_users()
                st.session_state["role"] = users_data.get(username.strip(), {}).get("role", "")
                # record login session
                try:
                    st.session_state["session_id"] = record_session_login(username.strip())
                except Exception:
                    st.session_state["session_id"] = None
                st.success("Logged in successfully.")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with right:
        st.subheader("Create account")

        # If no users exist, allow bootstrapping the first admin account
        if len(users) == 0:
            st.info("No accounts exist — create the initial admin account.")
            with st.form("register_form_bootstrap"):
                username = st.text_input("New username", key="register_username")
                password = st.text_input("New password", type="password", key="register_password")
                confirm = st.text_input("Confirm password", type="password", key="register_confirm")
                submitted = st.form_submit_button("Create admin account")

            if submitted:
                if password != confirm:
                    st.error("Passwords do not match.")
                else:
                            created, message = register_user(username, password, role="admin")
                            if created:
                                # persist to DB too
                                add_user(username.strip(), hash_password(password), "admin")
                                st.success(message)
                            else:
                                st.error(message)

        else:
            # Only allow admin users to create additional accounts
            if not st.session_state.get("authenticated") or st.session_state.get("role") != "admin":
                st.info("Only admin users can create new accounts. Please login as an admin.")
            else:
                with st.form("register_form"):
                    username = st.text_input("New username", key="register_username")
                    password = st.text_input("New password", type="password", key="register_password")
                    confirm = st.text_input("Confirm password", type="password", key="register_confirm")
                    role_choice = st.selectbox("Role", ["worker", "admin"], index=0)
                    submitted = st.form_submit_button("Create account")

                if submitted:
                    if password != confirm:
                        st.error("Passwords do not match.")
                    else:
                        created, message = register_user(username, password, role=role_choice)
                        if created:
                            add_user(username.strip(), hash_password(password), role_choice)
                            st.success(message)
                        else:
                            st.error(message)


def render_app() -> None:
    st.sidebar.success(f"Signed in as: {st.session_state['username']}")
    if st.sidebar.button("Logout"):
        # record logout session
        sid = st.session_state.get("session_id")
        if sid:
            try:
                record_session_logout(sid)
            except Exception:
                pass
        logout()
        st.rerun()

    if st.session_state.get("role") == "admin":
        st.title("Admin Dashboard")
        st.caption("Manage workers and review detection activity.")

        overview_tab, workers_tab, search_tab = st.tabs(["Overview", "Workers", "Search"])

        with overview_tab:
            try:
                workers = list_workers()
                total_workers = len(workers)
            except Exception:
                total_workers = 0

            try:
                rows = search_detections()
                total_dets = len(rows)
            except Exception:
                total_dets = 0

            stats = st.columns(3)
            stats[0].metric("Workers", total_workers)
            stats[1].metric("Detections", total_dets)
            stats[2].metric("Annotated", "Saved" if RUNS_DIR.exists() else "In memory")

            st.subheader("Recent detections")
            overview_rows = translate_grouped_detection_rows(group_detection_rows(search_detections()[:20]))
            st.dataframe(overview_rows, use_container_width=True)
            render_detection_previews(overview_rows, "Recent before/after previews", key_prefix="overview_preview")

        with workers_tab:
            st.subheader("Create worker")
            with st.form("admin_create_worker_form"):
                new_username = st.text_input("Worker username", key="admin_create_username")
                new_password = st.text_input("Worker password", type="password", key="admin_create_password")
                new_confirm = st.text_input("Confirm password", type="password", key="admin_create_confirm")
                submit_worker = st.form_submit_button("Create worker")

            if submit_worker:
                if new_password != new_confirm:
                    st.error("Passwords do not match.")
                elif not new_username or not new_password:
                    st.error("Username and password are required.")
                else:
                    created, message = register_user(new_username, new_password, role="worker")
                    if created:
                        try:
                            add_user(new_username.strip(), hash_password(new_password), "worker")
                        except Exception:
                            pass
                        st.success(f"Worker '{new_username}' created.")
                    else:
                        st.error(message)

            st.markdown("---")
            workers = list_workers()
            if workers:
                worker_names = [w["username"] for w in workers]
                st.write(f"Total workers: {len(worker_names)}")
                sel = st.selectbox("Select worker", ["-- choose --"] + worker_names, key="select_worker")
                col1, col2 = st.columns([2, 1])
                with col2:
                    st.write("Manage")
                    del_user = st.text_input("Delete selected worker", key="del_user")
                    if st.button("Delete worker"):
                        target = del_user.strip() or sel if sel != "-- choose --" else None
                        if not target:
                            st.error("Select or enter a worker username to delete")
                        else:
                            info = get_user(target)
                            if not info:
                                st.error("User not found")
                            elif info.get("role") != "worker":
                                st.error("Can only delete users with role=worker")
                            else:
                                if delete_user(target):
                                    st.success(f"Deleted worker {target}")
                                else:
                                    st.error("Failed to delete worker")

                if sel and sel != "-- choose --":
                    sessions = get_worker_sessions(sel)
                    st.write("Sessions (latest first):")
                    total_seconds = 0
                    rows = []
                    for s in sessions:
                        login = s.get("login_at")
                        logout_at = s.get("logout_at")
                        duration = "-"
                        if login and logout_at:
                            from datetime import datetime as _dt

                            try:
                                t0 = _dt.fromisoformat(login)
                                t1 = _dt.fromisoformat(logout_at)
                                delta = t1 - t0
                                duration = str(delta)
                                total_seconds += int(delta.total_seconds())
                            except Exception:
                                duration = "-"
                        rows.append({"id": s.get("id"), "login": login, "logout": logout_at, "duration": duration})
                    st.table(rows)
                    from datetime import timedelta

                    st.metric("Total work time", str(timedelta(seconds=total_seconds)))

                    products = get_worker_products(sel)
                    st.write("Products detected (by count):")
                    st.table(products)
            else:
                st.info("No workers yet.")

        with search_tab:
            st.subheader("Search detections")
            defect_types = list_defect_types()
            if not defect_types:
                defect_types = [
                    "Dry_joint",
                    "Incorrect_installation",
                    "PCB_damage",
                    "Short_circuit",
                ]
            defect_label_map = build_defect_label_map(defect_types)
            defect_labels = list(defect_label_map.keys())
            worker_search_tab, product_search_tab = st.tabs(["By worker", "By product"])

            with worker_search_tab:
                worker_q = st.selectbox(
                    "Select worker",
                    ["All workers"] + [w["username"] for w in list_workers()],
                    key="admin_worker_filter",
                )
                selected_defects = st.multiselect(
                    "Filter by defect types",
                    defect_labels,
                    default=[],
                    key="admin_worker_defect_filter",
                )
                product_q = st.text_input("Filter product name contains", key="admin_worker_product_filter")
                selected_defect_raw = [defect_label_map[label] for label in selected_defects]

                if worker_q == "All workers":
                    dets = search_detections(
                        class_names=selected_defect_raw or None,
                        product_name=product_q or None,
                    )
                else:
                    dets = get_worker_detections(
                        worker_q,
                        class_names=selected_defect_raw or None,
                        product_name=product_q or None,
                    )

                st.write(f"Found {len(dets)} detections")
                translated_dets = translate_grouped_detection_rows(group_detection_rows(dets))
                st.dataframe(translated_dets, use_container_width=True)
                render_detection_previews(translated_dets, "Worker detection previews", key_prefix=f"worker_preview_{worker_q}")

            with product_search_tab:
                product_q = st.text_input("Filter by product name", key="admin_product_search_name")
                selected_defects_product = st.multiselect(
                    "Filter by defect types",
                    defect_labels,
                    default=[],
                    key="admin_product_defect_filter",
                )
                selected_defect_product_raw = [defect_label_map[label] for label in selected_defects_product]
                dets = search_detections(
                    class_names=selected_defect_product_raw or None,
                    product_name=product_q or None,
                )
                st.write(f"Found {len(dets)} detections")
                translated_dets = translate_grouped_detection_rows(group_detection_rows(dets))
                st.dataframe(translated_dets, use_container_width=True)
                render_detection_previews(translated_dets, "Product detection previews", key_prefix="product_preview")
            return

    st.title("PCB Batch Test")

    st.caption("Upload multiple images, run YOLO inference once, and review predictions image by image.")
    controls = st.columns(4)
    with controls[0]:
        confidence_points = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90]
        conf = st.select_slider("Confidence threshold", options=confidence_points, value=0.05)
    with controls[1]:
        iou_points = [0.10, 0.20, 0.30, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90]
        iou = st.select_slider("IoU threshold", options=iou_points, value=0.45)
    with controls[2]:
        imgsz = st.selectbox("Image size", [640, 768, 960], index=0)
    with controls[3]:
        save_outputs = st.toggle("Save outputs to disk", value=True)

    uploaded_files = st.file_uploader(
        "Choose images",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
    )

    run_test = st.button("Run batch test", type="primary")

    if not run_test:
        return

    if not uploaded_files:
        st.warning("Please upload at least one image.")
        return

    model = get_model()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"{st.session_state['username']}_{timestamp}"
    if save_outputs:
        run_dir.mkdir(parents=True, exist_ok=True)
    # create run record
    run_id = create_run(st.session_state.get("username"))

    results = []
    progress = st.progress(0)
    detection_count = 0

    for index, uploaded_file in enumerate(uploaded_files, start=1):
        image_array = file_to_array(uploaded_file)
        result, detections = predict_one(
            image_array,
            model=model,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            save=False,
        )

        annotated_bgr = result.plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        original_path = run_dir / f"{Path(uploaded_file.name).stem}_original.png"
        annotated_path = run_dir / f"{Path(uploaded_file.name).stem}_annotated.png"
        save_rgb_image(image_array, original_path)
        cv2.imwrite(str(annotated_path), annotated_bgr)

        # persist detections
        for det in detections:
            add_detection(
                run_id,
                uploaded_file.name,
                Path(uploaded_file.name).stem,
                det.get("class_name"),
                det.get("confidence"),
                json.dumps(det.get("xyxy")),
                str(original_path),
                str(annotated_path),
            )
            detection_count += 1

        results.append(
            {
                "name": uploaded_file.name,
                "original": image_array,
                "annotated": annotated_rgb,
                "detections": translate_detection_rows(detections),
                "original_image_path": str(original_path),
                "annotated_image_path": str(annotated_path),
            }
        )
        progress.progress(index / len(uploaded_files))
    total_detections = sum(len(item["detections"]) for item in results)
    stats = st.columns(3)
    stats[0].metric("Images", len(results))
    stats[1].metric("Detections", total_detections)
    stats[2].metric("Annotated", "Saved" if save_outputs else "In memory")

    if save_outputs:
        st.info(f"Annotated images saved to: {run_dir}")

    # complete run record
    try:
        complete_run(run_id, len(uploaded_files), detection_count)
    except Exception:
        pass

    for item in results:
        with st.container(border=True):
            st.subheader(item["name"])
            left, right = st.columns(2)
            left.image(item["original"], caption="Original", use_container_width=True)
            right.image(item["annotated"], caption="Prediction", use_container_width=True)

            render_detection_summary(item)


def main() -> None:
    st.set_page_config(page_title="PCB Batch Tester", page_icon="📦", layout="wide")
    init_state()

    if st.session_state["authenticated"]:
        render_app()
    else:
        render_auth_screen()


if __name__ == "__main__":
    main()