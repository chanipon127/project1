from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
from datetime import datetime
import bcrypt
import os
import shutil
from fastapi.staticfiles import StaticFiles

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
    username: str = Form(...),        # เอามาจาก localStorage
    fullname: str = Form(None),
    password: str = Form(None),
    profile_img: UploadFile = File(None)
):
    try:
        with conn.cursor() as cur:
            updates = []
            values = []

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

        return {"message": "อัปเดตข้อมูลสำเร็จ"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาด: {str(e)}")

    