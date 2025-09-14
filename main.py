import json
from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
from datetime import datetime
import bcrypt
import os
import shutil
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
conn = psycopg2.connect(
    host="ep-floral-salad-a1wumcdl-pooler.ap-southeast-1.aws.neon.tech",
    database="neodb",
    user="neodb_owner",
    password="npg_8TuqdaBURE5Z",
    port=5432
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


# 🔹 ดึงกลุ่มการสอบไม่ซ้ำ
@app.get("/group_ids", response_model=List[str])
def get_group_ids():
    cursor = conn.cursor()
    cursor.execute('SELECT DISTINCT group_id FROM exam ORDER BY group_id')
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

# ----------------- Pydantic Model -----------------
class Answer(BaseModel):
    student_id: int
    group_id: str
    exam_year: int
    essay_text: str
    essay_analysis: str
    status: str


# -----------------------
# POST เพิ่มคำตอบ
@app.post("/api/answers")
def add_answer(answer: Answer):
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO answer (student_id, group_id, exam_year, essay_text, essay_analysis, status)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (answer.student_id, answer.group_id, answer.exam_year, answer.essay_text, answer.essay_analysis, answer.status))
        conn.commit()
        return {"message": "บันทึกคำตอบสำเร็จ"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()

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
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT answer_id, student_id, exam_year, essay_text, essay_analysis, group_id, status,score
            FROM answer
            ORDER BY answer_id DESC
        """)
        rows = cursor.fetchall()
        cursor.close()
        results = [
            {
                "answer_id": r[0],
                "student_id": r[1],
                "exam_year": r[2],
                "essay_text": r[3],
                "essay_analysis": r[4],
                "group_id": r[5],
                "status": r[6]
            }
            for r in rows
        ]
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ✅ API: ตรวจคำตอบด้วย AI
# -------------------------------
@app.post("/api/check-answer/{answer_id}")
async def check_answer(answer_id: int):
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT essay_text FROM answer WHERE answer_id = %s", (answer_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="ไม่พบคำตอบ")

        essay_text = row[0]

        # 🔹 ตรวจด้วย AI
        result = evaluate_single_answer(essay_text)

        if isinstance(result, dict):
            result_dict = result
        else:
            result_dict = json.loads(result)

        total_score = result_dict["คะแนนรวมทั้งหมด"]

        # 🔹 บันทึกลง DB 
        cursor.execute("""
            UPDATE answer 
            SET score=%s, status='ตรวจแล้ว'
            WHERE answer_id = %s
        """, (total_score, answer_id))
        conn.commit()

        return {"message": "ตรวจคำตอบสำเร็จ", "score": total_score}

    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()

