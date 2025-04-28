<?php
session_start();

// ตรวจสอบการเข้าสู่ระบบและสิทธิ์
if (!isset($_SESSION['username']) || !isset($_SESSION['role'])) {
    header("Location: login.php");
    exit();
}

$username = $_SESSION['username'];
$role = $_SESSION['role'];

if ($role !== 'admin') {
    echo "คุณไม่มีสิทธิ์เข้าถึงหน้านี้";
    exit();
}

// ดึงข้อมูลไว้ใช้ใน HTML
require 'admin_home.html';
?>