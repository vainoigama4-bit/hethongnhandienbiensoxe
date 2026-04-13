-- Bảng lưu thông tin xe trong bãi
CREATE TABLE ParkingEntry (
    PlateID NVARCHAR(20) PRIMARY KEY,
    DisplayPlate NVARCHAR(20),
    VehicleType NVARCHAR(10),
    Slot NVARCHAR(10),
    EntryTime DATETIME DEFAULT GETDATE()
);

-- Bảng thông tin khách hàng vé tháng
CREATE TABLE Customers (
    PlateID NVARCHAR(20) PRIMARY KEY,
    FullName NVARCHAR(100),
    Phone NVARCHAR(15),
    ExpiryDate DATETIME,
    IsMonthly BIT DEFAULT 1
);