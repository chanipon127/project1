import json
from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2,psycopg2.extras
from datetime import datetime
from fastapi.responses import StreamingResponse
import io, csv
import bcrypt
import os
import re
import shutil
import pandas as pd
import numpy as np
from fastapi import Query
from typing import List
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional
from ai_scoring import evaluate_single_answer


app = FastAPI()

# 🔓 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🌐 Database Connection
import psycopg2

conn = psycopg2.connect(
    host="ep-billowing-hall-a1n0l161-pooler.ap-southeast-1.aws.neon.tech",
    database="neondb",
    user="neondb_owner",
    password="npg_12pVAsiPLfxg",
    port=5432,
    sslmode="require"
)


# 📌 Schema
class RegisterForm(BaseModel):
    username: str
    fullname: str
    password: str
    role: str

# 🔐 Register
@app.post("/api/register")
async def register_user(data: RegisterForm):
    try:
        hashed_password = bcrypt.hashpw(data.password.encode('utf-8'), bcrypt.gensalt())
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO users (role, username, fullname, password, created_at)
            VALUES (%s, %s, %s, %s, %s)
        """, (data.role, data.username, data.fullname, hashed_password.decode('utf-8'), datetime.now()))
        conn.commit()
        return {"message": "สมัครสมาชิกสำเร็จ"}
    except Exception as e:
        conn.rollback()
        return {"message": f"เกิดข้อผิดพลาด: {str(e)}"}

# 📌 Schema
class LoginForm(BaseModel):
    username: str
    password: str
    
# ✅ Login
@app.post("/api/login")
async def login(data: LoginForm):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT password, role, fullname FROM users WHERE username = %s", (data.username,))
            result = cur.fetchone()
            if result:
                db_password, role, fullname = result
                if bcrypt.checkpw(data.password.encode('utf-8'), db_password.encode('utf-8')):
                    return {
                        "message": "เข้าสู่ระบบสำเร็จ",
                        "username": data.username,
                        "fullname": fullname,
                        "role": role
                    }
        raise HTTPException(status_code=401, detail="ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")
    
# 📌 Schema
class ContactForm(BaseModel):
    name: str
    user: str
    message: str

# ✉️ Contact Admin API
@app.post("/api/contact-admin")
async def admin_contact(data: ContactForm):
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO admin_contact (name, username, message, created_at)
            VALUES (%s, %s, %s, %s)
        """, (data.name, data.user, data.message, datetime.now()))
        conn.commit()
        return {"message": "ส่งข้อความถึงผู้ดูแลระบบเรียบร้อยแล้ว"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")

# 📂 ensure uploads folder
os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# 📌 Schema
class UserInfo(BaseModel):
    fullname: str
    username: str
    role: str
    profile_img: str = "https://via.placeholder.com/120"


# ✅ API: ดึงข้อมูลผู้ login
@app.get("/api/userinfo", response_model=UserInfo)
async def get_userinfo(username: str):
    cur = conn.cursor()
    cur.execute("SELECT fullname, role, profile_img FROM users WHERE username = %s", (username,))
    result = cur.fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="ไม่พบผู้ใช้")
    
    fullname, role, profile_img = result
    return {
        "fullname": fullname,
        "username": username,
        "role": role,
        "profile_img": profile_img or "https://via.placeholder.com/120"
    }

# ✅ Update User
@app.post("/api/update_user")
async def update_user(
    username: str = Form(...),  # เอามาจาก localStorage
    new_username: str = Form(None),   # username ใหม่
    fullname: str = Form(None),
    password: str = Form(None),
    profile_img: UploadFile = File(None)
):
    try:
        with conn.cursor() as cur:
            updates = []
            values = []

            if new_username:
                updates.append("username = %s")
                values.append(new_username)
                
            if fullname:
                updates.append("fullname = %s")
                values.append(fullname)

            if password:
                hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
                updates.append("password = %s")
                values.append(hashed_password)

            if profile_img:
                upload_dir = "uploads"
                os.makedirs(upload_dir, exist_ok=True)

                # ป้องกันชื่อไฟล์ชนกัน
                import uuid
                filename = f"{uuid.uuid4().hex}_{profile_img.filename}"
                file_path = os.path.join(upload_dir, filename)

                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(profile_img.file, buffer)

                file_url = f"http://127.0.0.1:8000/uploads/{filename}"
                updates.append("profile_img = %s")
                values.append(file_url)

            if not updates:
                raise HTTPException(status_code=400, detail="ไม่มีข้อมูลใหม่สำหรับอัปเดต")

            values.append(username)
            sql = f"UPDATE users SET {', '.join(updates)} WHERE username = %s"
            cur.execute(sql, tuple(values))

            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="ไม่พบผู้ใช้สำหรับอัปเดต")

            conn.commit()

        # ถ้าเปลี่ยน username ต้องอัปเดต localStorage ด้วย
        return {
            "message": "อัปเดตข้อมูลสำเร็จ",
            "new_username": new_username if new_username else username
        }
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")
    


# 📌 API: ดึงผู้ใช้งานทั้งหมด
@app.get("/api/users")
async def get_users():
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT username, fullname, role FROM users ORDER BY created_at DESC")
            rows = cur.fetchall()

        users = []
        for row in rows:
            username, fullname, role = row
            users.append({
                "username": username,
                "fullname": fullname,
                "role": role
            })

        return {"users": users}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")

#ลบผู้ใช้งาน
@app.delete("/api/delete_user")
async def delete_user(username: str):
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE username = %s", (username,))
            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="ไม่พบผู้ใช้งาน")
            conn.commit()
        return {"message": f"ลบผู้ใช้งาน {username} สำเร็จ"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")


# 🔹 ดึงปีการศึกษาไม่ซ้ำ
@app.get("/exam_years", response_model=List[int])
def get_exam_years():
    cursor = conn.cursor()
    cursor.execute('SELECT DISTINCT exam_year FROM exam ORDER BY exam_year DESC')
    years = [row[0] for row in cursor.fetchall()]
    return years


# 🔹 ดึงระดับชั้นเรียนไม่ซ้ำ
@app.get("/group_ids", response_model=List[str])
def get_group_ids():
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT group_id FROM exam
        UNION
        SELECT DISTINCT group_id FROM answer
        ORDER BY group_id
    """)
    groups = [row[0] for row in cursor.fetchall()]
    return groups

# 📌 ดึง feedback ทั้งหมด
@app.get("/api/contact-admin-all")
async def get_all_contacts(): 
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT contact_id, username, name, message, created_at
            FROM admin_contact
            ORDER BY created_at DESC
        """)
        rows = cursor.fetchall()
        cursor.close()
        # ✅ แปลง tuple → dict
        feedback_list = [
            {
                "contact_id": r[0],
                "username": r[1],
                "name": r[2],
                "message": r[3],
                "created_at": r[4]
            }
            for r in rows
        ]

        return feedback_list
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ ลบ feedback ------------------
@app.delete("/api/contact-admin/{contact_id}")
async def delete_feedback(contact_id: int):
    try:
        #conn = get_connection()
        cur = conn.cursor()
        # ตรวจสอบว่ามี contact_id นี้ไหม
        cur.execute("SELECT contact_id FROM admin_contact WHERE contact_id = %s", (contact_id,))
        if cur.fetchone() is None:
            cur.close()
            conn.close()
            raise HTTPException(status_code=404, detail="ไม่พบข้อความนี้")

        # ลบข้อความ
        cur.execute("DELETE FROM admin_contact WHERE contact_id = %s", (contact_id,))
        conn.commit()
        cur.close()
        conn.close()

        return {"message": "ลบข้อความเรียบร้อย"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# 📌 Schema (ใหม่)
class ContactForm(BaseModel):
    username: str
    message: str


# ✉️ Contact Admin API (ใหม่)
@app.post("/api/contact-admin")
async def admin_contact(data: ContactForm):
    try:
        cur = conn.cursor()

        # ✅ ดึง fullname จากตาราง users โดยใช้ username
        cur.execute("SELECT fullname FROM users WHERE username = %s", (data.username,))
        user_row = cur.fetchone()
        if not user_row:
            raise HTTPException(status_code=404, detail="ไม่พบผู้ใช้งานนี้")

        fullname = user_row[0]

        # ✅ บันทึกข้อความลงตาราง admin_contact
        cur.execute("""
            INSERT INTO admin_contact (name, username, message, created_at)
            VALUES (%s, %s, %s, %s)
        """, (fullname, data.username, data.message, datetime.now()))
        
        conn.commit()
        return {"message": f"ส่งข้อความถึงผู้ดูแลระบบเรียบร้อยแล้วโดย {fullname}"}
    
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")

# ----------------- Pydantic Model -----------------
class Answer(BaseModel):
    student_id: int
    group_id: str
    exam_year: int
    essay_text: str
    essay_analysis: str
    status: str



# -----------------------
# POST เพิ่มคำตอบ(เป็นไฟล์)
@app.post("/api/answers/upload")
async def upload_answers(file: UploadFile = File(...)):
    try:
        df = None
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file.file)
        elif file.filename.endswith(".xlsx"):
            df = pd.read_excel(file.file)
        else:
            raise HTTPException(status_code=400, detail="รองรับเฉพาะไฟล์ .csv หรือ .xlsx")

        cursor = conn.cursor()
        inserted = 0

        for _, row in df.iterrows():
            sid = row["student_id"]
            # ถ้าเป็น NaN ให้ข้าม
            if pd.isna(sid):
                continue

            # แปลงเป็น str และตัด .0 ถ้าเป็นตัวเลข
            if isinstance(sid, float) and sid.is_integer():
                student_id = str(int(sid))
            else:
                student_id = str(sid).strip()

            group_id = str(row["group_id"]).strip()

            exam_year_val = row.get("exam_year", None)
            if pd.isna(exam_year_val):
                continue  # ข้ามแถวที่ไม่มีปี
            exam_year = int(exam_year_val)

            # ✅ ตรวจสอบและเพิ่มปี/ชั้นในตาราง exam ถ้ายังไม่มี
            cursor.execute("""
                SELECT 1 FROM exam WHERE exam_year = %s AND group_id = %s
            """, (exam_year, group_id))
            if not cursor.fetchone():
                try:
                    cursor.execute("""
                        INSERT INTO exam (exam_year, group_id, exam_name, created_at)
                        VALUES (%s, %s, %s, %s)
                    """, (exam_year, group_id, "ภาษาไทย", datetime.now()))
                except psycopg2.errors.UniqueViolation:
                    conn.rollback()
                    cursor.execute("SELECT setval(pg_get_serial_sequence('exam', 'exam_id'), COALESCE(MAX(exam_id), 1), TRUE) FROM exam;")
                    conn.commit()

            # ✅ ตรวจซ้ำใน answer
            cursor.execute("""
                SELECT 1 FROM answer WHERE student_id=%s AND exam_year=%s AND group_id=%s
            """, (student_id, exam_year, group_id))
            if cursor.fetchone():
                continue

            # ✅ เพิ่ม answer
            cursor.execute("""
                INSERT INTO answer (student_id, group_id, exam_year, essay_text, essay_analysis, status)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                student_id,
                group_id,
                exam_year,
                row.get("essay_text", ""),
                row.get("essay_analysis", ""),
                "ยังไม่ได้ตรวจ"
            ))

            # ✅ ตรวจว่ามี teacher_score อยู่แล้วไหม
            cursor.execute("""
                SELECT 1 FROM teacher_score WHERE student_id=%s AND exam_year=%s AND group_id=%s
            """, (student_id, exam_year, group_id))

            if cursor.fetchone():
                # ✅ มีอยู่แล้ว → update คะแนนครูถ้ามีในไฟล์
                update_cols, values = [], []
                for i in range(1, 14):
                    for t in [1, 2]:
                        col = f"score_s{i}_t{t}"
                        if col in row and not pd.isna(row[col]):
                            update_cols.append(f"{col} = %s")
                            values.append(row[col])
                if update_cols:
                    values += [student_id, exam_year, group_id]
                    cursor.execute(f"""
                        UPDATE teacher_score SET {', '.join(update_cols)}
                        WHERE student_id=%s AND exam_year=%s AND group_id=%s
                    """, tuple(values))
            else:
                # ✅ ยังไม่มี → insert แถวใหม่
                score_cols_t1 = [f"score_s{i}_t1" for i in range(1, 14)]
                score_cols_t2 = [f"score_s{i}_t2" for i in range(1, 14)]
                score_cols = score_cols_t1 + score_cols_t2

                # ✅ อ่านค่าจากไฟล์ ถ้าไม่มีให้ใส่ None
                score_values = []
                for col in score_cols:
                    score_values.append(row[col] if col in row and not pd.isna(row[col]) else None)

                cursor.execute(f"""
                    INSERT INTO teacher_score (student_id, exam_year, group_id, {','.join(score_cols)})
                    VALUES (%s, %s, %s, {','.join(['%s']*len(score_cols))})
                """, [student_id, exam_year, group_id] + score_values)


            inserted+=1

        conn.commit()
        return {"message": "อัปโหลดสำเร็จ", "inserted": inserted}

    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cursor:
            cursor.close()




# ✅ โมเดลใหม่ รองรับคะแนนครูทั้งสองคน
class AnswerWithScore(BaseModel):
    student_id: int
    group_id: str
    exam_year: int
    essay_text: str
    essay_analysis: str
    status: str
    scores_t1: Optional[dict] = None  # {"s1":4, "s2":2, ...}
    scores_t2: Optional[dict] = None  # {"s1":3, "s2":2, ...}


# ✅ เพิ่มคำตอบ + คะแนนครู
@app.post("/api/answers")
async def create_answer(answer: AnswerWithScore):
    cursor = None
    try:
        cursor = conn.cursor()
        student_id_val = str(answer.student_id).strip()

        # 🔍 ตรวจว่ามีข้อมูลนี้ใน answer แล้วหรือยัง
        cursor.execute("""
            SELECT 1 FROM answer
            WHERE student_id=%s AND exam_year=%s AND group_id=%s
        """, (student_id_val, answer.exam_year, answer.group_id))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="คำตอบนี้มีอยู่แล้ว")

        # ✅ ตรวจสอบว่ามีปีการศึกษา/ชั้นเรียนในตาราง exam หรือยัง ถ้าไม่มีให้เพิ่ม
        cursor.execute("""
            SELECT 1 FROM exam WHERE exam_year = %s AND group_id = %s
        """, (answer.exam_year, answer.group_id))
        if not cursor.fetchone():
            cursor.execute("""
                INSERT INTO exam (exam_year, group_id, exam_name, created_at)
                VALUES (%s, %s, %s, %s)
            """, (answer.exam_year, answer.group_id, "ภาษาไทย", datetime.now()))

        # ✅ บันทึกคำตอบนักเรียน
        cursor.execute("""
            INSERT INTO answer (student_id, group_id, exam_year, essay_text, essay_analysis, status)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            student_id_val,
            answer.group_id,
            answer.exam_year,
            answer.essay_text,
            answer.essay_analysis,
            answer.status or "ยังไม่ได้ตรวจ"
        ))

        # ✅ เตรียมคอลัมน์คะแนนครู
        score_cols_t1 = [f"score_s{i}_t1" for i in range(1, 14)]
        score_cols_t2 = [f"score_s{i}_t2" for i in range(1, 14)]
        score_cols = score_cols_t1 + score_cols_t2

        # ✅ เตรียมค่าคะแนน
        values = []
        for i in range(1, 14):
            values.append(
                float(answer.scores_t1.get(f"s{i}", None))
                if answer.scores_t1 and f"s{i}" in answer.scores_t1 else None
            )
        for i in range(1, 14):
            values.append(
                float(answer.scores_t2.get(f"s{i}", None))
                if answer.scores_t2 and f"s{i}" in answer.scores_t2 else None
            )

        # ✅ บันทึกลงตาราง teacher_score
        cursor.execute(f"""
            INSERT INTO teacher_score (student_id, exam_year, group_id, {','.join(score_cols)})
            VALUES (%s, %s, %s, {','.join(['%s']*len(score_cols))})
        """, [student_id_val, answer.exam_year, answer.group_id] + values)

        conn.commit()
        return {"message": "เพิ่มคำตอบและคะแนนครูสำเร็จ ✅"}

    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")

    finally:
        if cursor:
            cursor.close()


# ✅ ดาวน์โหลด CSV เฉพาะคำตอบที่ตรวจแล้ว (ตามปีการศึกษาและระดับชั้น)
@app.get("/api/download-checked-csv")
async def download_checked_csv(exam_year: int, group_id: str):
    import re
    try:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cursor.execute("""
            SELECT exam_year, group_id, student_id, score, description
            FROM answer
            WHERE status = 'ตรวจแล้ว' AND exam_year = %s AND group_id = %s
            ORDER BY student_id
        """, (exam_year, group_id))
        rows = cursor.fetchall()

        output = io.StringIO()
        writer = csv.writer(output)
        header = ["exam_year", "group_id", "student_id", "score"] + [f"s{i}_score" for i in range(1, 14)]
        writer.writerow(header)

        for row in rows:
            s_scores = [""] * 13
            desc_raw = row.get("description")

            if desc_raw:
                try:
                    # ✅ บางฐานข้อมูล description อาจเป็น str หรือ dict
                    desc_json = desc_raw if isinstance(desc_raw, dict) else json.loads(desc_raw)

                    for i in range(1, 14):
                        key = f"s{i}"
                        if key not in desc_json:
                            continue
                        item = desc_json[key]

                        # เริ่มจากชั้นแรก
                        score_value = item.get("score", "")
                        fb = item.get("feedback")

                        # ✅ ตรวจชนิด feedback
                        fb_json = None
                        if isinstance(fb, str):
                            try:
                                fb_json = json.loads(fb)
                            except Exception:
                                # บางครั้ง feedback เป็นข้อความปกติ ไม่ใช่ JSON
                                fb_json = None
                        elif isinstance(fb, dict):
                            fb_json = fb

                        # ✅ พยายามดึงคะแนนจาก feedback ถ้ามี
                        if fb_json:
                            for k in ["คะแนนรวม", "คะแนนรวมใจความ", "score_total", "score"]:
                                if k in fb_json and isinstance(fb_json[k], (int, float)):
                                    score_value = fb_json[k]
                                    break

                        # ✅ ถ้ายังไม่มีคะแนน ลองค้นหาใน string JSON
                        if (score_value == "" or score_value is None) and isinstance(fb, str):
                            match = re.search(r'"score"\s*:\s*([0-9.]+)', fb)
                            if match:
                                score_value = float(match.group(1))

                        s_scores[i - 1] = score_value
                except Exception as e:
                    print("⚠️ parse error:", e)

            writer.writerow([
                row["exam_year"],
                row["group_id"],
                row["student_id"],
                row.get("score", ""),
                *s_scores
            ])

        output.seek(0)
        headers = {
            "Content-Disposition": f"attachment; filename=checked_scores_{group_id}_{exam_year}.csv",
            "Content-Type": "text/csv"
        }
        return StreamingResponse(iter([output.getvalue()]), headers=headers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




# GET ดึงคำตอบทั้งหมด
@app.get("/api/answers-all")
def get_all_answers():
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT answer_id, student_id, exam_year, essay_text, essay_analysis, group_id, status
            FROM answer
            ORDER BY answer_id DESC
        """)
        rows = cursor.fetchall()
        cursor.close()
        results = []
        for r in rows:
            results.append({
                "answer_id": r[0],
                "student_id": r[1],
                "exam_year": r[2],
                "essay_text": r[3],
                "essay_analysis": r[4],
                "group_id": r[5],
                "status": r[6]
            })
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()


# 🔹 Pydantic Model
# -------------------------------
class Answer(BaseModel):
    student_id: int
    group_id: str
    exam_year: int
    essay_text: str
    essay_analysis: str
    status: str

# -------------------------------
# ✅ API: ดึงคำตอบทั้งหมด
# -------------------------------
@app.get("/api/answers-all")
def get_all_answers():
    cursor = None
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT answer_id, student_id, exam_year, essay_text, essay_analysis, group_id, status, score
            FROM answer
            ORDER BY answer_id DESC
        """)
        rows = cursor.fetchall()

        results = []
        for r in rows:
            results.append({
                "answer_id": r[0],
                "student_id": r[1],
                "exam_year": r[2],
                "essay_text": r[3],
                "essay_analysis": r[4],
                "group_id": r[5],
                "status": r[6],
                "score": r[7] if r[7] is not None else None
            })
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cursor:
            cursor.close()


# ✅ API: ตรวจคำตอบด้วย AI
# ✅ API: ตรวจคำตอบด้วย AI (เวอร์ชันสุดท้ายที่แก้ไขตาม Log จริง)
@app.post("/api/check-answer/{answer_id}")
async def check_answer(answer_id: int):
    cursor = None
    try:
        cursor = conn.cursor()
        # ✅ 1. ดึงคำตอบนักเรียนจากฐานข้อมูล
        cursor.execute("SELECT essay_text, essay_analysis FROM answer WHERE answer_id = %s", (answer_id,))
        row = cursor.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="ไม่พบคำตอบ")

        essay_text, essay_analysis = row

        # 1. เรียก AI และรับผลลัพธ์เป็น Dictionary
        ai_result_dict = evaluate_single_answer(essay_text, essay_analysis)
        # Log ผลดิบไว้เผื่อตรวจสอบในอนาคต
        print("AI raw result for answer_id", answer_id, ":", json.dumps(ai_result_dict, indent=2, ensure_ascii=False))

        # 2. ฟังก์ชันแปลงผลลัพธ์สำหรับโครงสร้าง JSON แบบ "แบน" (Flat)
        def map_ai_results_to_s_format(results):
            formatted_desc = {}

            # *** สร้าง Mapping ของ Key ที่ถูกต้อง 100% จาก Log ***
            # หมายเหตุ: s3 มี key ที่ผิดปกติจาก AI แต่เราต้องใช้ตามนั้น
            key_mapping = {
                "s1": "ข้อที่ 1 - ใจความสำคัญ",
                "s2": "ข้อที่ 1 - การเรียงลำดับและเชื่อมโยงความคิด",
                "s3": "ข้อที่ 1 - ความถูกต้องตามหลักการเขียนย่อความ",
                "s4": "ข้อที่ 1 - การสะกดคำ",
                "s5": "ข้อที่ 1 - การใช้คำ/ถ้อยคำสำนวน",
                "s6": "ข้อที่ 1 - การใช้ประโยค",
                "s7": "ข้อที่ 2 - คำบอกข้อคิดเห็น",
                "s8": "ข้อที่ 2 - เหตุผลสนับสนุน",
                "s9": "ข้อที่ 2 - การเรียงลำดับและเชื่อมโยงความคิด",
                "s10": "ข้อที่ 2 - ความถูกต้องตามหลักการแสดงความคิดเห็น",
                "s11": "ข้อที่ 2 - การสะกดคำ/การใช้ภาษา",
                "s12": "ข้อที่ 2 - การใช้คำ/ถ้อยคำสำนวน",
                "s13": "ข้อที่ 2 - การใช้ประโยค",
            }
            
            # วน Loop เพื่อดึงค่าของแต่ละ S
            for s_key, ai_key in key_mapping.items():
                # ดึงข้อมูลจาก top-level dictionary โดยตรง
                data = results.get(ai_key, {}) 
                
                # ✅ S1 ใช้ key "คะแนนรวมใจความ"
                if s_key == "s1":
                    score = data.get("คะแนนรวมใจความ", 0.0)
                else:
                    score = data.get("score", data.get("คะแนน", 0.0))
                
                # ใช้ 'details' เป็น feedback ถ้ามี, ถ้าไม่มีก็ใช้ object ทั้งหมด
                feedback_data = data.get("details", data)
                
                formatted_desc[s_key] = {
                    "score": float(score),
                    "feedback": json.dumps(feedback_data, ensure_ascii=False)
                }
            
            # ดึงคะแนนรวมทั้งหมด
            total_score = results.get("คะแนนรวมทั้งหมด", 0.0)
            return formatted_desc, float(total_score)

        # 3. เรียกใช้ฟังก์ชันแปลงค่า
        formatted_description, total_score = map_ai_results_to_s_format(ai_result_dict)
        
        # 4. บันทึกลงฐานข้อมูล
        cursor.execute("""
            UPDATE answer
            SET score=%s,
                status='ตรวจแล้ว',
                description=%s
            WHERE answer_id = %s
        """, (total_score, json.dumps(formatted_description, ensure_ascii=False), answer_id))
        
        conn.commit()

        return {
            "message": "ตรวจคำตอบสำเร็จ",
            "score": total_score,
            "description": formatted_description
        }

    except Exception as e:
        if conn:
            conn.rollback()
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")
    finally:
        if cursor:
            cursor.close()




# -----------------------------
# API: ดูผลคำตอบ + คะแนนครู (เวอร์ชันปรับปรุง)
# -----------------------------
@app.get("/api/view-score/{answer_id}")
def view_score(answer_id: int):
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # ดึงคำตอบหลัก
        cur.execute("""
            SELECT answer_id, student_id, group_id, exam_year,
                   essay_text, essay_analysis, status, score, description
            FROM answer
            WHERE answer_id = %s
        """, (answer_id,))
        answer = cur.fetchone()
        if not answer:
            raise HTTPException(status_code=404, detail="ไม่พบคำตอบนี้")

        # ดึงคะแนนครู (teacher_score)
        cur.execute("""
            SELECT * FROM teacher_score
            WHERE student_id = %s AND exam_year = %s AND group_id = %s
        """, (answer["student_id"], answer["exam_year"], answer["group_id"]))
        teacher_row = cur.fetchone()

        teacher_scores = {"teacher1": {}, "teacher2": {}}
        if teacher_row:
            for i in range(1, 14):
                teacher_scores["teacher1"][f"s{i}"] = teacher_row.get(f"score_s{i}_t1")
                teacher_scores["teacher2"][f"s{i}"] = teacher_row.get(f"score_s{i}_t2")

        # ✅ ส่งออกเป็น JSON
        return {
            "answer_id": answer["answer_id"],
            "student_id": answer["student_id"],
            "group_id": answer["group_id"],
            "exam_year": answer["exam_year"],
            "essay_text": answer["essay_text"],
            "essay_analysis": answer["essay_analysis"],
            "status": answer["status"],
            "score": answer["score"],
            "description": answer["description"],   # <- JSON ที่มี score/feedback ของ AI
            "teacher_scores": teacher_scores
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))