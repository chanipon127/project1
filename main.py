from fastapi import FastAPI, Request
from pydantic import BaseModel
import psycopg2
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
import bcrypt


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ✨ เชื่อม PostgreSQL (ใส่ข้อมูลของคุณ)
conn = psycopg2.connect(
    host="ep-floral-salad-a1wumcdl-pooler.ap-southeast-1.aws.neon.tech",
    database="neodb",
    user="neodb_owner",
    password="npg_8TuqdaBURE5Z",
    port=5432
)
cursor = conn.cursor()

# 🔹 สร้าง schema ของ request
class RegisterForm(BaseModel):
    username: str
    email: str
    password: str
    role: str

@app.post("/api/register")
async def register_user(data: RegisterForm):
    try:
        # เข้ารหัสรหัสผ่านก่อนเก็บ
        hashed_password = bcrypt.hashpw(data.password.encode('utf-8'), bcrypt.gensalt())
        
        cursor.execute(
            """
            INSERT INTO "users" (role, username, email, password, created_at)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (data.role, data.username, data.email, hashed_password.decode('utf-8'), datetime.now())
        )
        conn.commit()
        return {"message": "สมัครสมาชิกสำเร็จ"}
    except Exception as e:
        conn.rollback()
        return {"message": f"เกิดข้อผิดพลาด: {str(e)}"}

from fastapi import HTTPException

class LoginForm(BaseModel):
    username: str
    password: str

@app.post("/api/login")
async def login(data: LoginForm):
    try:
        # ดึงรหัสผ่านจาก DB ตาม username
        cursor.execute(
            "SELECT password, role FROM users WHERE username = %s",
            (data.username,)
        )
        result = cursor.fetchone()
        if result:
            db_password, role = result
            # ตรวจสอบรหัสผ่าน
            if bcrypt.checkpw(data.password.encode('utf-8'), db_password.encode('utf-8')):
                return {"message": "เข้าสู่ระบบสำเร็จ", "role": role}
        
        # ถ้าไม่ตรง
        raise HTTPException(status_code=401, detail="ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดภายในเซิร์ฟเวอร์: {str(e)}")

