import psycopg2

try:
    conn = psycopg2.connect(
        host="ep-floral-salad-a1wumcdl-pooler.ap-southeast-1.aws.neon.tech",
        database="neodb",
        user="neodb_owner",
        password="npg_8TuqdaBURE5Z",
        port=5432,
        sslmode="require"
    )
    print("✅ Connected successfully")
except Exception as e:
    print("❌ Error:", e)
