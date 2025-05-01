from fastapi import FastAPI, Request
from pydantic import BaseModel
import psycopg2
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

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
        cursor.execute(
            """
            INSERT INTO "user" (role,username,email, password, created_at)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (data.role,data.username, data.email, data.password, datetime.now())
        )
        conn.commit()
        return {"message": "สมัครสมาชิกสำเร็จ"}
    except Exception as e:
        conn.rollback()
        return {"message": f"เกิดข้อผิดพลาด: {str(e)}"}
