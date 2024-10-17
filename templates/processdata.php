<?php
$servername = "localhost";
$username = "hokengakari";
$password = "230073";
$dbname = "shoubou_data";

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $weather = $_POST['weather'];
    $year_month = $_POST['year_month'];
    $age = $_POST['age'];
    $place = $_POST['place'];
    $gender = $_POST['gender'];
    $age_group = $_POST['age_group'];

    $sql = "INSERT INTO your_table_name (覚知年月日, 覚知曜日, 天候, 出場場所地区, 性別, 年齢区分_サーベイランス用)
    VALUES ('$weather', '$year_month', '$age', '$place', '$gender', '$age_group')";

    if ($conn->query($sql) === TRUE) {
        echo "New record created successfully";
    } else {
        echo "Error: " . $sql . "<br>" . $conn->error;
    }
}

$conn->close();
?>
