import os, cv2, sqlite3, datetime, uuid, pickle, csv, time
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# ---------- CONFIG ----------
DB_PATH ="attendance.db"
EXIT_TIMEOUT_SEC =30
FACE_SIM_THRESH =0.55
REG_IMAGE_COUNT =5
DEVICE_ID ="webcam"

# ---------- DATABASE ----------
def new_id():
    return str(uuid.uuid4())

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript(open("migrate.sql").read())
    conn.commit()
    conn.close()

def add_employee(emp_id, full_name, email, dept, emp_no):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO employees VALUES (?,?,?,?,?,?)", (emp_id, full_name, email, dept, emp_no, datetime.date.today()))
    conn.commit()
    conn.close()

def employee_id_by_name(full_name):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    row = cur.execute("SELECT employee_id FROM employees WHERE full_name = ?", (full_name,)).fetchone()
    conn.close()
    return row[0] if row else None

def save_template(emp_id, embedding, conf):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    emb_blob = pickle.dumps(embedding, protocol=pickle.HIGHEST_PROTOCOL)
    cur.execute("INSERT INTO face_templates VALUES (?,?,?,?)", (new_id(), emp_id, emb_blob, conf))
    conn.commit()
    conn.close()

def load_all_templates():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    rows = cur.execute("SELECT employee_id, embedding FROM face_templates").fetchall()
    conn.close()
    return [(emp_id, pickle.loads(emb)) for emp_id, emb in rows]

def log_attendance(emp_id, event_type, conf):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO attendance_logs VALUES (?,?,?,?,?,?)", (new_id(), emp_id, event_type, datetime.datetime.now(), DEVICE_ID, conf))
    conn.commit()
    conn.close()

def fetch_between(start, end):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    rows = cur.execute("SELECT employee_id, event_type, timestamp, confidence_score FROM attendance_logs WHERE DATE(timestamp) BETWEEN ? AND ? ORDER BY timestamp", (start, end)).fetchall()
    conn.close()
    return rows

def generate_reports():
    today = datetime.date.today()
    periods = [
        ("daily", today, today),
        ("weekly", today - datetime.timedelta(days=today.weekday()), today),
        ("monthly", today.replace(day=1), today)
    ]
    for prefix, start, end in periods:
        rows = fetch_between(start, end)
        fname = f"{prefix}_{today}.csv"
        import pandas as pd
        pd.DataFrame(rows, columns=["employee_id", "event_type", "timestamp", "confidence_score"]).to_csv(fname, index=False)
        print(f"📊{fname} created")

# ---------- REGISTRATION ----------
def register_user():
    full_name = input("Full name: ").strip().lower()
    email = input("Email: ").strip()
    dept = input("Department: ").strip()
    emp_no = input("Employee no: ").strip()
    emp_id = new_id()
    add_employee(emp_id, full_name, email, dept, emp_no)
    folder = f"data/{full_name}"
    os.makedirs(folder, exist_ok=True)
    cap = cv2.VideoCapture(0)
    count, embeddings = 0, []
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0)
    while count < REG_IMAGE_COUNT:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Registration", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            cv2.imwrite(f"{folder}/{count:02d}.jpg", gray)
            faces = app.get(frame)  # Use the original color frame for face detection
            if faces:
                embeddings.append(faces[0]['embedding'])
                count += 1
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    if embeddings:
        save_template(emp_id, np.mean(embeddings, axis=0), 0.99)
        print("✅ Registered:", full_name)
    else:
        print("❌ No faces detected.")

# ---------- LIVE ATTENDANCE ----------
def live_attendance():
    templates = load_all_templates()
    if not templates:
        print("No templates. Register first.")
        return
    known_vecs = np.array([emb for _, emb in templates])
    known_ids = [emp_id for emp_id, _ in templates]
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0)
    last_seen = {}
    cap = cv2.VideoCapture(0)
    print("Live attendance running. ESC to stop.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = app.get(frame)
        current = set()
        for face in faces:
            bbox = face['bbox'].astype(int)
            emb = face['embedding'].reshape(1, -1)
            sims = cosine_similarity(emb, known_vecs)[0]
            idx = sims.argmax()
            if sims[idx] >= FACE_SIM_THRESH:
                emp_id = known_ids[idx]
                current.add(emp_id)
                cv2.rectangle(frame, tuple(bbox[:2]), tuple(bbox[2:]), (0,255,0), 2)
                cv2.putText(frame, emp_id[:6], (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        now = time.time()
        for emp_id in current:
            if emp_id not in last_seen:
                log_attendance(emp_id, 'ENTRY', sims[idx])
                print(f"{datetime.datetime.now():%H:%M:%S} ENTRY {emp_id}")
                last_seen[emp_id] = now
        for emp_id in list(last_seen.keys()):
            if now - last_seen[emp_id] > EXIT_TIMEOUT_SEC:
                log_attendance(emp_id, 'EXIT', 1.0)
                print(f"{datetime.datetime.now():%H:%M:%S} EXIT {emp_id}")
                del last_seen[emp_id]
        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

# ---------- CLI ----------
def main():
    init_db()
    while True:
        print("\n1.Register 2.Train 3.Attendance 4.Reports 5.Exit")
        choice = input("Select: ").strip()
        if choice == "1":
            register_user()
        elif choice == "2":
            live_attendance() # training auto-happens during registration
        elif choice == "3":
            live_attendance()
        elif choice == "4":
            generate_reports()
        elif choice == "5":
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()
