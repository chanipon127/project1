from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
from datetime import datetime
import bcrypt

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
    
# 🔐 Login
@app.post("/api/login")
async def login(data: LoginForm):
    try:
        cur = conn.cursor()
        cur.execute("SELECT password, role FROM users WHERE username = %s", (data.username,))
        result = cur.fetchone()
        if result:
            db_password, role = result
            if bcrypt.checkpw(data.password.encode('utf-8'), db_password.encode('utf-8')):
                return {"message": "เข้าสู่ระบบสำเร็จ", "role": role}
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
