<?php
session_start();

// ตรวจสอบว่า user ยังไม่ได้เข้าสู่ระบบ
if (!isset($_SESSION['username'])) {
    header("Location: login.php"); // ถ้ายังไม่ได้เข้าสู่ระบบจะเปลี่ยนเส้นทางไปที่หน้า login
    exit();
}

$username = $_SESSION['username'];
$role = $_SESSION['role'];
?>

<!DOCTYPE html>
<html lang="th">
<head>
  <meta charset="UTF-8">
  <title>หน้าหลัก</title>
</head>
<body>
  <h1>ยินดีต้อนรับ, <?php echo $username; ?>!</h1>
  <p>คุณเป็นผู้ตรวจข้อสอบประเภท: <?php echo $role; ?></p>

  <a href="logout.php">ออกจากระบบ</a>
</body>
</html>