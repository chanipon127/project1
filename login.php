<?php
session_start();

// ข้อมูลการเชื่อมต่อ Neon Database (PostgreSQL)
$host = 'ep-floral-salad-a1wumcdl-pooler.ap-southeast-1.aws.neon.tech';  // แก้เป็น host ของคุณ
$dbname = 'neodb';
$user = 'neodb_owner';
$password = 'npg_8TuqdaBURE5Z';
$sslmode = 'require'; // Neon ต้องใช้ SSL

// รับค่าจากฟอร์ม
$username = $_POST['username'] ?? '';
$password_input = $_POST['password'] ?? '';

try {
    // สร้างการเชื่อมต่อ
    $dsn = "pgsql:host=$host;dbname=$dbname;sslmode=$sslmode";
    $pdo = new PDO($dsn, $user, $password, [
        PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION
    ]);

    // ค้นหาผู้ใช้ในฐานข้อมูล
    $stmt = $pdo->prepare("SELECT * FROM users WHERE username = :username");
    $stmt->execute(['username' => $username]);
    $user = $stmt->fetch(PDO::FETCH_ASSOC);

    // ตรวจสอบรหัสผ่าน
    if ($user && password_verify($password_input, $user['password'])) {
        // เข้าสู่ระบบสำเร็จ
        $_SESSION['username'] = $user['username'];
        $_SESSION['role'] = $user['role'];
        echo "<script>alert('เข้าสู่ระบบสำเร็จ'); window.location.href = 'dashboard.php';</script>";
    } else {
        // เข้าสู่ระบบไม่สำเร็จ
        echo "<script>alert('ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง'); window.history.back();</script>";
    }

} catch (PDOException $e) {
    echo "เกิดข้อผิดพลาดในการเชื่อมต่อฐานข้อมูล: " . $e->getMessage();
}
?>
