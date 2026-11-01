from flask import (Flask, Response, jsonify, render_template, request,
                   redirect, url_for, session, flash)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps
from ultralytics import YOLO
import cv2, os, threading, time, sqlite3, json, uuid
from traffic_logic import SignalController

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY")
if not app.secret_key:
    raise RuntimeError("Set the FLASK_SECRET_KEY environment variable before starting the app.")

folder = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(folder, "database.db")
YOLO_MODEL = os.path.join(folder, "yolov8n.pt")
AMBULANCE_MODEL = os.path.join(folder, "best.pt")
DEFAULT_VIDEO = os.path.join(folder, "traffic.mp4")

NUM_LANES = 4
VEHICLES = {2, 3, 5, 7}
controller = SignalController(NUM_LANES)
state = {
    "counts": [0] * NUM_LANES,
    "green": 0,
    "time_left": max(0, int(controller.green_until - time.time())),
    "ambulance": False,
    "status": "waiting",
    "error": "Waiting for video processing to start.",
    "video": os.path.basename(DEFAULT_VIDEO),
}
latest_frame = None
lock = threading.Lock()
video_lock = threading.Lock()
worker_lock = threading.Lock()
current_video = {"path": DEFAULT_VIDEO, "version": 0}
detection_thread = None
UPLOAD_DIR = os.path.join(folder, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def init_db():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT NOT NULL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS traffic_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, counts TEXT, green_lane INTEGER, ambulance INTEGER)""")
    con.commit()
    con.close()


def log_to_db(counts, green, amb):
    con = sqlite3.connect(DB)
    con.execute("INSERT INTO traffic_log (ts, counts, green_lane, ambulance) VALUES (?,?,?,?)",
                (time.strftime("%H:%M:%S"), json.dumps(counts), green, int(amb)))
    con.commit()
    con.close()


def update_state(**values):
    with lock:
        state.update(values)


def ensure_detection_worker():
    global detection_thread
    with worker_lock:
        if detection_thread is None or not detection_thread.is_alive():
            update_state(status="starting", error="Loading detection models.")
            detection_thread = threading.Thread(target=detection_loop, daemon=True)
            detection_thread.start()


def detection_loop():
    global latest_frame
    model = YOLO(YOLO_MODEL)
    amb_model = YOLO(AMBULANCE_MODEL)
    frame_no, last_amb, last_log = 0, 0, 0
    cap = None
    loaded_version = -1

    while True:
        with video_lock:
            video_path = current_video["path"]
            video_version = current_video["version"]

        if video_version != loaded_version:
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(video_path)
            loaded_version = video_version
            frame_no, last_amb, last_log = 0, 0, 0
            if not cap.isOpened():
                cap.release()
                cap = None
                update_state(
                    status="error",
                    error=f"Could not open video: {os.path.basename(video_path)}",
                    video=os.path.basename(video_path),
                )
                time.sleep(1)
                continue
            update_state(
                status="processing",
                error=None,
                video=os.path.basename(video_path),
                counts=[0] * NUM_LANES,
                green=controller.green_lane,
                time_left=max(0, int(controller.green_until - time.time())),
                ambulance=False,
            )

        if cap is None:
            time.sleep(1)
            continue

        ok, frame = cap.read()
        if not ok:
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            position = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if frame_count > 0 and position < frame_count - 1:
                cap.release()
                cap = None
                update_state(
                    status="error",
                    error=f"Video decoding failed: {os.path.basename(video_path)}",
                )
                time.sleep(1)
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            if not ok:
                cap.release()
                cap = None
                update_state(
                    status="error",
                    error=f"Video contains no readable frames: {os.path.basename(video_path)}",
                )
                time.sleep(1)
                continue

        with video_lock:
            if current_video["version"] != video_version:
                continue

        frame_no += 1
        frame = cv2.resize(frame, (960, 540))
        h, w = frame.shape[:2]
        lane_w = w // NUM_LANES
        counts = [0] * NUM_LANES

        for box in model(frame, verbose=False)[0].boxes:
            if int(box.cls[0]) not in VEHICLES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            lane = min(((x1 + x2) // 2) // lane_w, NUM_LANES - 1)
            counts[lane] += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if frame_no % 3 == 0:
            for b in amb_model(frame, verbose=False)[0].boxes:
                if float(b.conf[0]) >= 0.7:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    lane = min(((x1 + x2) // 2) // lane_w, NUM_LANES - 1)
                    controller.emergency(lane)
                    last_amb = time.time()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, "AMBULANCE", (x1, max(y1 - 8, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        green, left = controller.update(counts)

        for i in range(NUM_LANES):
            x = i * lane_w
            cv2.line(frame, (x, 0), (x, h), (255, 0, 0), 2)
            color = (0, 200, 0) if i == green else (0, 0, 255)
            cv2.putText(frame, f"Lane {i+1}: {counts[i]}", (x + 10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        amb_active = (time.time() - last_amb) < 15
        if time.time() - last_log >= 5:
            log_to_db(counts, green, amb_active)
            last_log = time.time()

        _, buf = cv2.imencode(".jpg", frame)
        with video_lock:
            if current_video["version"] != video_version:
                continue
        with lock:
            latest_frame = buf.tobytes()
            state.update(
                counts=counts,
                green=green,
                time_left=left,
                ambulance=amb_active,
                status="processing",
                error=None,
            )


def stream():
    while True:
        with lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.03)


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        if not username or len(password) < 4:
            flash("Enter a username and a password of at least 4 characters.")
            return render_template("register.html")
        con = sqlite3.connect(DB)
        first = con.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
        role = "admin" if first else "user"   # first account becomes admin
        try:
            con.execute("INSERT INTO users (username, password, role) VALUES (?,?,?)",
                        (username, generate_password_hash(password), role))
            con.commit()
        except sqlite3.IntegrityError:
            flash("Username already exists.")
            con.close()
            return render_template("register.html")
        con.close()
        flash("Account created. Please log in.")
        return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        con = sqlite3.connect(DB)
        row = con.execute("SELECT password, role FROM users WHERE username=?",
                          (request.form["username"].strip(),)).fetchone()
        con.close()
        if row and check_password_hash(row[0], request.form["password"]):
            session["user"] = request.form["username"].strip()
            session["role"] = row[1]
            return redirect(url_for("index"))
        flash("Wrong username or password.")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    ensure_detection_worker()
    return render_template("dashboard.html", lanes=NUM_LANES)


@app.route("/analysis")
@login_required
def analysis():
    if session.get("role") != "admin":
        return "Admins only.", 403
    return render_template("analysis.html")


@app.route("/video-feed")
@login_required
def video_feed():
    return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/stats")
@login_required
def stats():
    with lock:
        return jsonify(state)


@app.route("/api/history")
@login_required
def history():
    con = sqlite3.connect(DB)
    rows = con.execute("SELECT ts, counts, ambulance FROM traffic_log ORDER BY id DESC LIMIT 60").fetchall()
    con.close()
    rows.reverse()
    return jsonify([{"ts": r[0], "counts": json.loads(r[1]), "ambulance": r[2]} for r in rows])

@app.route("/upload", methods=["POST"])
@login_required
def upload():
    global latest_frame
    f = request.files.get("video")
    if not f or not f.filename:
        flash("Choose a video file to upload.", "error")
        return redirect(url_for("index"))

    filename = secure_filename(f.filename)
    extension = os.path.splitext(filename)[1].lower()
    if extension not in (".mp4", ".avi", ".mov"):
        flash("Unsupported video format. Upload an MP4, AVI, or MOV file.", "error")
        return redirect(url_for("index"))

    path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{extension}")
    f.save(path)
    test_cap = cv2.VideoCapture(path)
    valid = test_cap.isOpened()
    ok, frame = test_cap.read() if valid else (False, None)
    test_cap.release()
    if not valid or not ok or frame is None or frame.size == 0:
        if os.path.exists(path):
            os.remove(path)
        flash("The uploaded file could not be decoded as a video. Choose a valid video and try again.", "error")
        return redirect(url_for("index"))

    with video_lock:
        current_video["path"] = path
        current_video["version"] += 1
    with lock:
        latest_frame = None
        state.update(
            counts=[0] * NUM_LANES,
            green=controller.green_lane,
            time_left=max(0, int(controller.green_until - time.time())),
            ambulance=False,
            status="restarting",
            error=None,
            video=os.path.basename(path),
        )
    ensure_detection_worker()
    flash("Video validated. Detection is restarting with the uploaded video.", "success")
    return redirect(url_for("index"))


@app.route("/report.csv")
@login_required
def report():
    if session.get("role") != "admin":
        return "Admins only.", 403
    con = sqlite3.connect(DB)
    rows = con.execute("SELECT ts, counts, green_lane, ambulance FROM traffic_log").fetchall()
    con.close()
    lines = ["time,lane1,lane2,lane3,lane4,green_lane,ambulance"]
    for ts, c, g, a in rows:
        lines.append(f"{ts},{','.join(map(str, json.loads(c)))},{g + 1},{a}")
    return Response("\n".join(lines), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=traffic_report.csv"})
if __name__ == "__main__":
    init_db()
    threading.Thread(target=detection_loop, daemon=True).start()
    app.run(debug=False, threaded=True)