<?php
session_start();

$host = "localhost";
$dbname = "ai_thai";
$username = "root";
$password = "";

$conn = new mysqli($host, $username, $password, $dbname);
if ($conn->connect_error) {
    die("เชื่อมต่อฐานข้อมูลล้มเหลว: " . $conn->connect_error);
}

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $input_username = $_POST['username'];
    $input_password = $_POST['password'];

    // ตรวจสอบในตาราง admin ก่อน
    $sql_admin = "SELECT * FROM admin WHERE username = ?";
    $stmt = $conn->prepare($sql_admin);
    $stmt->bind_param("s", $input_username);
    $stmt->execute();
    $result = $stmt->get_result();

    if ($result->num_rows > 0) {
        $user = $result->fetch_assoc();
        if (password_verify($input_password, $user['password'])) {
            $_SESSION['username'] = $input_username;
            $_SESSION['user_id'] = $user['admin_id'];
            $_SESSION['role'] = 'admin';
            header("Location: homepage_admin.html");
            exit();  // ปิดการประมวลผลหลังจากเปลี่ยนเส้นทาง
        } else {
            // ถ้ารหัสผ่านไม่ถูกต้อง
            header("Location: login.html?error=invalid_credentials");
            exit();
        }
    } else {
        // ถ้าไม่พบใน admin ให้ลองตรวจสอบใน examiners
        $sql_examiner = "SELECT * FROM examiners WHERE username = ?";
        $stmt = $conn->prepare($sql_examiner);
        $stmt->bind_param("s", $input_username);
        $stmt->execute();
        $result = $stmt->get_result();

        if ($result->num_rows > 0) {
            $examiner = $result->fetch_assoc();
            if (password_verify($input_password, $examiner['password'])) {
                $_SESSION['username'] = $input_username;
                $_SESSION['user_id'] = $examiner['examiner_id'];
                $_SESSION['full_name'] = $examiner['full_name'];
                $_SESSION['group_id'] = $examiner['group_id'];
                $_SESSION['role'] = 'examiner';
                header("Location: homepage_user.html");
                exit();
            } else {
                // ถ้ารหัสผ่านไม่ถูกต้อง
                header("Location: login.html?error=invalid_credentials");
                exit();
            }
        } else {
            // ถ้าไม่พบชื่อผู้ใช้ในทั้งสองตาราง
            header("Location: login.html?error=invalid_credentials");
            exit();
        }
    }
}

$conn->close();
?>