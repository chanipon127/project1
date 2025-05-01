<?php
$host = "localhost";
$dbname = "ai_thai";
$username = "root";
$password = "";

$conn = new mysqli($host, $username, $password, $dbname);
if ($conn->connect_error) {
    die("เชื่อมต่อฐานข้อมูลล้มเหลว: " . $conn->connect_error);
}

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $username = $_POST['username'];
    $email = $_POST['email'];
    $plain_password = $_POST['password'];
    $role = $_POST['role'];

    // ตรวจสอบชื่อผู้ใช้หรืออีเมลซ้ำ
    $check_sql = "SELECT * FROM user WHERE username = ? OR email = ?";
    $stmt = $conn->prepare($check_sql);
    $stmt->bind_param("ss", $username, $email);
    $stmt->execute();
    $result = $stmt->get_result();

    if ($result->num_rows > 0) {
        echo "ชื่อผู้ใช้หรืออีเมลนี้ถูกใช้ไปแล้ว";
    } else {
        // เข้ารหัสรหัสผ่าน
        $hashed_password = password_hash($plain_password, PASSWORD_DEFAULT);

        // บันทึกข้อมูล
        $sql = "INSERT INTO user (username, email, password, role) VALUES (?, ?, ?, ?)";
        $stmt = $conn->prepare($sql);
        $stmt->bind_param("ssss", $username, $email, $hashed_password, $role);

        if ($stmt->execute()) {
            echo "สมัครสมาชิกสำเร็จ! <a href='login.html'>เข้าสู่ระบบ</a>";
        } else {
            echo "เกิดข้อผิดพลาด: " . $stmt->error;
        }
    }

    $stmt->close();
}

$conn->close();
?>
