from flask import (Flask, Response, jsonify, render_template, request,
                   redirect, url_for, session, flash)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps
from ultralytics import YOLO
import cv2, os, threading, time, sqlite3, json
from traffic_logic import SignalController

app = Flask(__name__)
app.secret_key = "change-this-to-any-random-text"
folder = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(folder, "database.db")

NUM_LANES = 4
VEHICLES = {2, 3, 5, 7}
controller = SignalController(NUM_LANES)
state = {"counts": [0] * NUM_LANES, "green": 0, "time_left": 0, "ambulance": False}
latest_frame = None
lock = threading.Lock()
current_video = {"path": os.path.join(folder, "traffic.mp4"), "changed": False}
UPLOAD_DIR = os.path.join(folder, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
current_video = {"path": os.path.join(folder, "traffic.mp4"), "changed": False}


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


def detection_loop():
    global latest_frame
    cap = cv2.VideoCapture(current_video["path"])
    if not cap.isOpened():
        print("ERROR: could not open traffic.mp4")
        return
    model = YOLO("yolov8n.pt")
    amb_model = YOLO(os.path.join(folder, "best.pt"))
    frame_no, last_amb, last_log = 0, 0, 0

    while True:
        if current_video["changed"]:
            cap.release()
            cap = cv2.VideoCapture(current_video["path"])
            current_video["changed"] = False
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
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
        if time.time() - last_log >= 5:   # save a record every 5 seconds
            log_to_db(counts, green, amb_active)
            last_log = time.time()

        _, buf = cv2.imencode(".jpg", frame)
        with lock:
            latest_frame = buf.tobytes()
            state.update(counts=counts, green=green, time_left=left, ambulance=amb_active)


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
    f = request.files.get("video")
    if f and f.filename.lower().endswith((".mp4", ".avi", ".mov")):
        path = os.path.join(UPLOAD_DIR, secure_filename(f.filename))
        f.save(path)
        current_video["path"] = path
        current_video["changed"] = True
    return redirect(url_for("index"))


@app.route("/report.csv")
@login_required
def report():
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