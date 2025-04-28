<?php
$host = "localhost";
$dbname = "ai_thai";
$username = "root";  // หรือชื่อ user ของคุณ
$password = "";      // ถ้ามีรหัสผ่านให้ใส่

$conn = new mysqli($host, $username, $password, $dbname);

if ($conn->connect_error) {
    die("เชื่อมต่อฐานข้อมูลล้มเหลว: " . $conn->connect_error);
}
?>