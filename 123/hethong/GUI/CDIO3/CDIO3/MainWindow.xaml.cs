using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Data;
using System.Data.SqlClient;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace CDIO3
{
    public partial class MainWindow : Window
    {
        // ========== CONFIG ==========
        private const string API_RECOGNIZE_URL = "http://localhost:5000/api/recognize";
        private const string API_BARRIER_URL = "http://localhost:5000/api/barrier";
        private const string CONN_STRING = @"Data Source=(LocalDB)\MSSQLLocalDB;AttachDbFilename=D:\Code\C#\CDIO3\CDIO3\ParkingDatabase.mdf;Integrated Security=True;";

        private const string IMG_ENTRY_FOLDER = "D:\\Code\\CDIO3\\imgcar\\entry";
        private const string IMG_EXIT_FOLDER = "D:\\Code\\CDIO3\\imgcar\\exit";

        // folder ảnh mô phỏng (ảnh input)
        private const string SIM_ENTRY_SOURCE_FOLDER = "D:\\Code\\CDIO3\\sim_img\\entry";
        private const string SIM_EXIT_SOURCE_FOLDER = "D:\\Code\\CDIO3\\sim_img\\exit";

        private const bool SIM_ENABLED = true;

        // Giá vé
        private const decimal PRICE_PER_HOUR_BIKE = 5000m;
        private const decimal PRICE_PER_HOUR_CAR = 15000m;

        // Khách tháng demo
        private readonly Dictionary<string, CustomerInfo> customerDatabase = new Dictionary<string, CustomerInfo>();

        // Danh sách ảnh mô phỏng
        private List<string> simEntryImages = new List<string>();
        private List<string> simExitImages = new List<string>();
        private readonly Random rng = new Random();

        // Ảnh đang hiển thị
        private string currentSimEntryImagePath = null;
        private string currentSimExitImagePath = null;

        // Ảnh đã inject vào imgcar (để API nhận)
        private string lastInjectedEntryPath = null;
        private string lastInjectedExitPath = null;

        // Cache kết quả nhận diện (để Confirm không gọi API lần 2)
        private RecognitionResult lastEntryRecognition = null;
        private RecognitionResult lastExitRecognition = null;

        // Cache preview thanh toán (để Confirm Exit chỉ xác nhận)
        private ExitPaymentPreview lastExitPaymentPreview = null;

        private class ExitPaymentPreview
        {
            public string NormalizedPlate { get; set; }
            public string DisplayPlate { get; set; }
            public VehicleEntry Entry { get; set; }
            public DateTime ExitTime { get; set; }
            public TimeSpan Duration { get; set; }
            public decimal Amount { get; set; }
            public bool IsFree { get; set; }
        }

        public MainWindow()
        {
            InitializeComponent();

            InitializeFolders();
            InitializeDatabase();

            if (SIM_ENABLED)
                InitializeSimulation();

            Loaded += MainWindow_Loaded;

            ResetEntryUIWaiting();
            ResetExitPaymentUI();
            UpdateStatusBar("Hệ thống đang chạy...");
        }

        // ========== INIT ==========
        private void InitializeFolders()
        {
            try
            {
                Directory.CreateDirectory(IMG_ENTRY_FOLDER);
                Directory.CreateDirectory(IMG_EXIT_FOLDER);

                if (SIM_ENABLED)
                {
                    Directory.CreateDirectory(SIM_ENTRY_SOURCE_FOLDER);
                    Directory.CreateDirectory(SIM_EXIT_SOURCE_FOLDER);
                }

                Console.WriteLine("Folders initialized");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Lỗi tạo thư mục: {ex.Message}", "Lỗi", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void InitializeDatabase()
        {
            customerDatabase.Add("29A12345", new CustomerInfo
            {
                Name = "NGUYỄN VĂN A",
                Phone = "0901234567",
                ExpiryDate = new DateTime(2026, 12, 31),
                IsMonthlyCustomer = true
            });

            customerDatabase.Add("30H67890", new CustomerInfo
            {
                Name = "TRẦN THỊ B",
                Phone = "0912345678",
                ExpiryDate = new DateTime(2026, 6, 30),
                IsMonthlyCustomer = true
            });

            customerDatabase.Add("51F12345", new CustomerInfo
            {
                Name = "LÊ VĂN C",
                Phone = "0923456789",
                ExpiryDate = new DateTime(2025, 12, 31),
                IsMonthlyCustomer = true
            });

            Console.WriteLine($"Database initialized with {customerDatabase.Count} customers");
        }

        private void InitializeSimulation()
        {
            simEntryImages = GetAllImages(SIM_ENTRY_SOURCE_FOLDER);
            simExitImages = GetAllImages(SIM_EXIT_SOURCE_FOLDER);

            Console.WriteLine($"Simulation images loaded: entry={simEntryImages.Count}, exit={simExitImages.Count}");
        }

        private async void MainWindow_Loaded(object sender, RoutedEventArgs e)
        {
            if (!SIM_ENABLED) return;

            await ShowNextEntryImageAsync();
            await ShowNextExitImageAsync();
        }

        // ========== IMAGE SOURCE ==========
        private List<string> GetAllImages(string folder)
        {
            try
            {
                if (!Directory.Exists(folder)) return new List<string>();

                var exts = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
                { ".jpg", ".jpeg", ".png", ".bmp" };

                return Directory.EnumerateFiles(folder)
                    .Where(f => exts.Contains(Path.GetExtension(f)))
                    .ToList();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"GetAllImages error: {ex.Message}");
                return new List<string>();
            }
        }

        private async Task ShowNextEntryImageAsync()
        {
            if (simEntryImages == null || simEntryImages.Count == 0)
                simEntryImages = GetAllImages(SIM_ENTRY_SOURCE_FOLDER);

            if (simEntryImages.Count == 0)
            {
                txtEntryPlate.Text = "No entry images";
                txtEntryPlate.Foreground = Brushes.Gray;
                return;
            }

            currentSimEntryImagePath = simEntryImages[rng.Next(simEntryImages.Count)];
            LoadImage(currentSimEntryImagePath, imgEntryCamera);

            ResetEntryUIWaiting();

            await AutoRecognizeEntryFromCurrentImage();
        }

        private async Task ShowNextExitImageAsync()
        {
            if (simExitImages == null || simExitImages.Count == 0)
                simExitImages = GetAllImages(SIM_EXIT_SOURCE_FOLDER);

            if (simExitImages.Count == 0)
            {
                txtExitPlate.Text = "No exit images";
                txtExitPlate.Foreground = Brushes.Gray;
                return;
            }

            currentSimExitImagePath = simExitImages[rng.Next(simExitImages.Count)];
            LoadImage(currentSimExitImagePath, imgExitCamera);

            txtExitPlate.Text = "Đang nhận diện...";
            txtExitPlate.Foreground = Brushes.Gray;

            ResetExitPaymentUI();

            await AutoRecognizeExitFromCurrentImage();
        }

        // ========== AUTO RECOGNIZE ==========
        private async Task AutoRecognizeEntryFromCurrentImage()
        {
            lastEntryRecognition = null;
            lastInjectedEntryPath = null;

            if (string.IsNullOrEmpty(currentSimEntryImagePath) || !File.Exists(currentSimEntryImagePath))
                return;

            lastInjectedEntryPath = CopyAsNewImage(currentSimEntryImagePath, IMG_ENTRY_FOLDER, "entry");
            if (string.IsNullOrEmpty(lastInjectedEntryPath))
            {
                txtEntryPlate.Text = "Copy ảnh lỗi";
                txtEntryPlate.Foreground = Brushes.Gray;
                return;
            }

            var result = await RecognizePlateFromFile(lastInjectedEntryPath, "entry");
            if (result != null && result.Success)
            {
                lastEntryRecognition = result;

                txtEntryPlate.Text = result.LicensePlate;
                txtEntryPlate.Foreground = result.Confidence >= 80 ? Brushes.Green : Brushes.Orange;

                LoadImage(result.ImagePath, imgEntryCamera);
            }
            else
            {
                txtEntryPlate.Text = "Không nhận diện";
                txtEntryPlate.Foreground = Brushes.Gray;
            }
        }

        /// <summary>
        /// Auto nhận diện EXIT -> tự truy vấn SQL + tính tiền + show UI (preview).
        /// Nút confirm exit chỉ bật khi preview OK.
        /// </summary>
        private async Task AutoRecognizeExitFromCurrentImage()
        {
            lastExitRecognition = null;
            lastInjectedExitPath = null;

            ResetExitPaymentUI();

            if (string.IsNullOrEmpty(currentSimExitImagePath) || !File.Exists(currentSimExitImagePath))
                return;

            lastInjectedExitPath = CopyAsNewImage(currentSimExitImagePath, IMG_EXIT_FOLDER, "exit");
            if (string.IsNullOrEmpty(lastInjectedExitPath))
            {
                txtExitPlate.Text = "Copy ảnh lỗi";
                txtExitPlate.Foreground = Brushes.Gray;
                return;
            }

            var result = await RecognizePlateFromFile(lastInjectedExitPath, "exit");
            if (result != null && result.Success)
            {
                lastExitRecognition = result;

                await Dispatcher.InvokeAsync(() =>
                {
                    txtExitPlate.Text = result.LicensePlate;
                    txtExitPlate.Foreground = result.Confidence >= 80 ? Brushes.Green : Brushes.Orange;
                    LoadImage(result.ImagePath, imgExitCamera);
                });

                // PREVIEW thanh toán (tính tiền + show UI)
                await PreviewExitPaymentAsync(result);
            }
            else
            {
                txtExitPlate.Text = "Không nhận diện";
                txtExitPlate.Foreground = Brushes.Gray;
                ResetExitPaymentUI();
            }
        }

        // ========== CONFIRM BUTTONS ==========
        private async void BtnConfirmEntry_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                if (lastEntryRecognition == null || !lastEntryRecognition.Success)
                {
                    MessageBox.Show("Chưa có kết quả nhận diện hợp lệ để xác nhận.", "Info",
                        MessageBoxButton.OK, MessageBoxImage.Information);
                    return;
                }

                await ProcessEntryVehicle(lastEntryRecognition);
                await ShowNextEntryImageAsync();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        /// <summary>
        /// Confirm EXIT: chỉ được bấm sau khi PreviewExitPaymentAsync chạy OK (đã hiện tiền).
        /// Confirm sẽ xoá ParkingEntry + mở barrier.
        /// </summary>
        private async void BtnConfirmExit_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                if (lastExitPaymentPreview == null)
                {
                    MessageBox.Show("Chưa có thông tin thanh toán. Vui lòng đợi hệ thống tính tiền trước.", "Info",
                        MessageBoxButton.OK, MessageBoxImage.Information);
                    return;
                }

                SetExitConfirmEnabled(false); // chống double click

                // (demo) lưu record thanh toán
                SavePaymentRecord(
                    lastExitPaymentPreview.NormalizedPlate,
                    lastExitPaymentPreview.Entry,
                    lastExitPaymentPreview.ExitTime,
                    lastExitPaymentPreview.Amount
                );

                // Xoá record khỏi ParkingEntry sau khi xác nhận thanh toán
                DeleteParkingEntryFromSQL(lastExitPaymentPreview.NormalizedPlate);

                // Mở barrier
                await SendBarrierCommand("exit", "open");

                UpdateStatusBar(
                    $"Xe ra: {lastExitPaymentPreview.DisplayPlate} - Thu phí: " +
                    $"{(lastExitPaymentPreview.IsFree ? "Miễn phí" : lastExitPaymentPreview.Amount.ToString("N0") + " VNĐ")}"
                );

                // Chuyển ảnh xe ra tiếp theo
                await ShowNextExitImageAsync();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Error", MessageBoxButton.OK, MessageBoxImage.Error);

                // nếu lỗi thì bật lại nếu vẫn còn preview
                SetExitConfirmEnabled(lastExitPaymentPreview != null);
            }
        }

        // ========== API ==========
        private async Task<RecognitionResult> RecognizePlateFromFile(string imagePath, string gate)
        {
            try
            {
                using (HttpClient client = new HttpClient())
                {
                    client.Timeout = TimeSpan.FromSeconds(15);

                    using (var form = new MultipartFormDataContent())
                    {
                        byte[] imageData = File.ReadAllBytes(imagePath);
                        form.Add(new ByteArrayContent(imageData), "image", Path.GetFileName(imagePath));
                        form.Add(new StringContent(gate), "gate");

                        var response = await client.PostAsync(API_RECOGNIZE_URL, form);
                        if (!response.IsSuccessStatusCode)
                        {
                            Console.WriteLine($"API HTTP error: {response.StatusCode}");
                            return null;
                        }

                        string json = await response.Content.ReadAsStringAsync();
                        var apiResult = JsonConvert.DeserializeObject<ApiRecognitionResponse>(json);

                        if (apiResult != null && apiResult.Success)
                        {
                            return new RecognitionResult
                            {
                                Success = true,
                                LicensePlate = apiResult.LicensePlate,
                                Confidence = apiResult.Confidence,
                                ImagePath = imagePath,
                                Timestamp = DateTime.Now
                            };
                        }

                        Console.WriteLine($"API returned success=false. error={apiResult?.Error}");
                        return null;
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Recognition API error: {ex.Message}");
                return null;
            }
        }

        private async Task SendBarrierCommand(string gate, string action)
        {
            try
            {
                using (HttpClient client = new HttpClient())
                {
                    string url = $"{API_BARRIER_URL}/{gate}/{action}";
                    var response = await client.PostAsync(url, null);

                    if (response.IsSuccessStatusCode)
                        Console.WriteLine($"Barrier command sent: {gate} {action}");
                    else
                        Console.WriteLine($"Barrier command failed: {response.StatusCode}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to send barrier command: {ex.Message}");
            }
        }

        // ========== BUSINESS LOGIC ==========
        private async Task ProcessEntryVehicle(RecognitionResult result)
        {
            await Dispatcher.InvokeAsync(() =>
            {
                string normalizedPlate = NormalizeLicensePlate(result.LicensePlate);

                txtEntryPlate.Text = result.LicensePlate;
                txtEntryPlate.Foreground = result.Confidence >= 80 ? Brushes.Green : Brushes.Orange;
                LoadImage(result.ImagePath, imgEntryCamera);

                if (customerDatabase.TryGetValue(normalizedPlate, out var customer))
                {
                    DisplayCustomerInfo(customer);

                    if (customer.ExpiryDate < DateTime.Now)
                    {
                        txtCustomerStatus.Text = "VÉ THÁNG HẾT HẠN";
                        brdCustomerStatus.Background = new SolidColorBrush(Color.FromRgb(255, 200, 0));
                    }
                    else
                    {
                        txtCustomerStatus.Text = "KHÁCH HÀNG VÉ THÁNG";
                        brdCustomerStatus.Background = new SolidColorBrush(Color.FromRgb(76, 175, 80));
                    }
                }
                else
                {
                    txtCustomerStatus.Text = "KHÁCH VÃNG LAI";
                    brdCustomerStatus.Background = new SolidColorBrush(Color.FromRgb(224, 224, 224));
                    HideCustomerInfo();
                }

                AutoDetectVehicleType(result.LicensePlate);
                txtEntrySlot.Text = GenerateParkingSlot();

                string vehicleType = cbEntryVehicleType.SelectedIndex == 0 ? "Xe máy" : "Ô tô";

                SaveVehicleEntryToSQLUpsert(
                    normalizedPlate: normalizedPlate,
                    displayPlate: result.LicensePlate,
                    vehicleType: vehicleType,
                    slot: txtEntrySlot.Text,
                    entryTime: DateTime.Now
                );

                _ = SendBarrierCommand("entry", "open");
                UpdateStatusBar($"Xe vào: {result.LicensePlate} - Slot: {txtEntrySlot.Text}");
            });
        }

        /// <summary>
        /// Tính tiền + show UI (không xoá DB, không mở barrier).
        /// </summary>
        private async Task PreviewExitPaymentAsync(RecognitionResult result)
        {
            string normalizedPlate = NormalizeLicensePlate(result.LicensePlate);

            lastExitPaymentPreview = null;
            SetExitConfirmEnabled(false);

            VehicleEntry entry = GetParkingEntryFromSQL(normalizedPlate);

            if (entry == null)
            {
                await Dispatcher.InvokeAsync(() =>
                {
                    lblTimeIn.Text = "--/--/---- --:--";
                    lblTotalHour.Text = "--";
                    lblAmount.Text = "KHÔNG TÌM THẤY";
                    lblAmount.Foreground = Brushes.Gray;
                });

                UpdateStatusBar($"Không tìm thấy thông tin xe: {result.LicensePlate} -> chuyển ảnh khác");

                await ShowNextExitImageAsync();
                return;
            }

            DateTime exitTime = DateTime.Now;
            TimeSpan duration = exitTime - entry.EntryTime;

            decimal pricePerHour = entry.VehicleType == "Xe máy" ? PRICE_PER_HOUR_BIKE : PRICE_PER_HOUR_CAR;

            decimal totalHours = (decimal)duration.TotalHours;
            if (totalHours < 1) totalHours = 1;

            decimal amount = Math.Ceiling(totalHours) * pricePerHour;

            bool isFree = false;
            if (customerDatabase.TryGetValue(normalizedPlate, out var customer))
            {
                if (customer.IsMonthlyCustomer && customer.ExpiryDate >= DateTime.Now)
                {
                    amount = 0;
                    isFree = true;
                }
            }

            await Dispatcher.InvokeAsync(() =>
            {
                lblTimeIn.Text = entry.EntryTime.ToString("dd/MM/yyyy HH:mm");
                lblTotalHour.Text = FormatDuration(duration);

                lblAmount.Text = isFree ? "MIỄN PHÍ (VÉ THÁNG)" : $"{amount:N0} VNĐ";
                lblAmount.Foreground = isFree ? Brushes.Green : Brushes.Red;
            });

            lastExitPaymentPreview = new ExitPaymentPreview
            {
                NormalizedPlate = normalizedPlate,
                DisplayPlate = result.LicensePlate,
                Entry = entry,
                ExitTime = exitTime,
                Duration = duration,
                Amount = amount,
                IsFree = isFree
            };

            SetExitConfirmEnabled(true);
        }

        // ========== SQL ==========
        private VehicleEntry GetParkingEntryFromSQL(string normalizedPlate)
        {
            try
            {
                using (SqlConnection conn = new SqlConnection(CONN_STRING))
                {
                    conn.Open();
                    string query = "SELECT PlateID, DisplayPlate, VehicleType, Slot, EntryTime FROM ParkingEntry WHERE PlateID = @Plate";

                    using (SqlCommand cmd = new SqlCommand(query, conn))
                    {
                        cmd.Parameters.AddWithValue("@Plate", normalizedPlate);

                        using (SqlDataReader reader = cmd.ExecuteReader(CommandBehavior.SingleRow))
                        {
                            if (!reader.Read()) return null;

                            DateTime entryTime = DateTime.Now;
                            if (reader["EntryTime"] != DBNull.Value)
                                entryTime = Convert.ToDateTime(reader["EntryTime"]);

                            return new VehicleEntry
                            {
                                LicensePlate = reader["PlateID"]?.ToString(),
                                DisplayPlate = reader["DisplayPlate"]?.ToString(),
                                VehicleType = reader["VehicleType"]?.ToString(),
                                Slot = reader["Slot"]?.ToString(),
                                EntryTime = entryTime
                            };
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"GetParkingEntryFromSQL error: {ex.Message}");
                return null;
            }
        }

        private void DeleteParkingEntryFromSQL(string normalizedPlate)
        {
            try
            {
                using (SqlConnection conn = new SqlConnection(CONN_STRING))
                {
                    conn.Open();
                    string query = "DELETE FROM ParkingEntry WHERE PlateID = @Plate";

                    using (SqlCommand cmd = new SqlCommand(query, conn))
                    {
                        cmd.Parameters.AddWithValue("@Plate", normalizedPlate);
                        cmd.ExecuteNonQuery();
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"DeleteParkingEntryFromSQL error: {ex.Message}");
            }
        }

        private void SaveVehicleEntryToSQLUpsert(string normalizedPlate, string displayPlate, string vehicleType, string slot, DateTime entryTime)
        {
            try
            {
                using (SqlConnection conn = new SqlConnection(CONN_STRING))
                {
                    conn.Open();

                    string query = @"
IF EXISTS (SELECT 1 FROM ParkingEntry WHERE PlateID = @Plate)
BEGIN
    UPDATE ParkingEntry
    SET DisplayPlate = @DisplayPlate,
        VehicleType = @VehicleType,
        Slot = @Slot,
        EntryTime = @EntryTime
    WHERE PlateID = @Plate
END
ELSE
BEGIN
    INSERT INTO ParkingEntry (PlateID, DisplayPlate, VehicleType, Slot, EntryTime)
    VALUES (@Plate, @DisplayPlate, @VehicleType, @Slot, @EntryTime)
END";

                    using (SqlCommand cmd = new SqlCommand(query, conn))
                    {
                        cmd.Parameters.AddWithValue("@Plate", normalizedPlate);
                        cmd.Parameters.AddWithValue("@DisplayPlate", (object)displayPlate ?? DBNull.Value);
                        cmd.Parameters.AddWithValue("@VehicleType", (object)vehicleType ?? DBNull.Value);
                        cmd.Parameters.AddWithValue("@Slot", (object)slot ?? DBNull.Value);
                        cmd.Parameters.AddWithValue("@EntryTime", entryTime);
                        cmd.ExecuteNonQuery();
                    }

                    Console.WriteLine($"Đã upsert vào SQL: {displayPlate}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"SaveVehicleEntryToSQLUpsert error: {ex.Message}");
            }
        }

        // ========== UI HELPERS ==========
        private void SetExitConfirmEnabled(bool enabled)
        {
            btnConfirmExit.IsEnabled = enabled;
            btnConfirmExit.Opacity = enabled ? 1.0 : 0.6;
        }

        // ========== HELPERS ==========
        private string CopyAsNewImage(string sourcePath, string destFolder, string gate)
        {
            try
            {
                if (string.IsNullOrEmpty(sourcePath) || !File.Exists(sourcePath))
                    return null;

                string ext = Path.GetExtension(sourcePath);
                if (string.IsNullOrEmpty(ext)) ext = ".jpg";

                string fileName = $"{gate}_{DateTime.Now:yyyyMMdd_HHmmss_fff}{ext}";
                string destPath = Path.Combine(destFolder, fileName);

                File.Copy(sourcePath, destPath, overwrite: true);
                return destPath;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"CopyAsNewImage error: {ex.Message}");
                return null;
            }
        }

        private void LoadImage(string imagePath, System.Windows.Controls.Image imageControl)
        {
            try
            {
                if (!string.IsNullOrEmpty(imagePath) && File.Exists(imagePath))
                {
                    BitmapImage bitmap = new BitmapImage();
                    bitmap.BeginInit();
                    bitmap.CacheOption = BitmapCacheOption.OnLoad;
                    bitmap.CreateOptions = BitmapCreateOptions.IgnoreImageCache;
                    bitmap.UriSource = new Uri(Path.GetFullPath(imagePath));
                    bitmap.EndInit();
                    bitmap.Freeze();
                    imageControl.Source = bitmap;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to load image: {ex.Message}");
            }
        }

        private string NormalizeLicensePlate(string plate)
        {
            return plate?.Replace("-", "").Replace(" ", "").Replace(".", "").ToUpper() ?? "";
        }

        private void AutoDetectVehicleType(string licensePlate)
        {
            string normalized = NormalizeLicensePlate(licensePlate);

            if (normalized.Length >= 3)
            {
                string prefix = normalized.Substring(0, 2);
                if (int.TryParse(prefix, out _) && char.IsLetter(normalized[2]))
                {
                    cbEntryVehicleType.SelectedIndex = 1; // Ô tô
                    return;
                }
            }

            cbEntryVehicleType.SelectedIndex = 0; // Xe máy
        }

        private void DisplayCustomerInfo(CustomerInfo customer)
        {
            pnlCustomerInfo.Visibility = Visibility.Visible;
            lblCustomerName.Text = customer.Name;
            lblCustomerPhone.Text = $"SĐT: {customer.Phone}";
            lblExpiryDate.Text = $"Hạn dùng: {customer.ExpiryDate:dd/MM/yyyy}";
            lblExpiryDate.Foreground = customer.ExpiryDate < DateTime.Now ? Brushes.Red : Brushes.Green;
        }

        private void HideCustomerInfo()
        {
            pnlCustomerInfo.Visibility = Visibility.Collapsed;
        }

        private void ResetEntryUIWaiting()
        {
            txtEntryPlate.Text = "Đang nhận diện...";
            txtEntryPlate.Foreground = Brushes.Gray;

            txtCustomerStatus.Text = "ĐANG CHỜ XE...";
            brdCustomerStatus.Background = new SolidColorBrush(Color.FromRgb(224, 224, 224));

            HideCustomerInfo();
            txtEntrySlot.Text = "";
            cbEntryVehicleType.SelectedIndex = 0;
        }

        private void ResetExitPaymentUI()
        {
            lblTimeIn.Text = "--/--/---- --:--";
            lblTotalHour.Text = "--";
            lblAmount.Text = "--";
            lblAmount.Foreground = Brushes.Gray;

            lastExitPaymentPreview = null;
            SetExitConfirmEnabled(false);
        }

        private void SavePaymentRecord(string licensePlate, VehicleEntry entry, DateTime exitTime, decimal amount)
        {
            Console.WriteLine($"Payment recorded: Plate={entry.DisplayPlate ?? licensePlate}, Type={entry.VehicleType}, Amount={amount:N0}, Exit={exitTime:dd/MM/yyyy HH:mm}");
        }

        private string GenerateParkingSlot()
        {
            string[] zones = { "A", "B", "C", "D" };
            string zone = zones[rng.Next(zones.Length)];
            int number = rng.Next(101, 199);
            return $"{zone}-{number}";
        }

        private string FormatDuration(TimeSpan duration)
        {
            if (duration.TotalHours < 1)
                return $"{duration.Minutes} phút";
            return $"{(int)duration.TotalHours} giờ {duration.Minutes} phút";
        }

        private void UpdateStatusBar(string message)
        {
            Console.WriteLine($"[{DateTime.Now:HH:mm:ss}] {message}");
            Title = $"Hệ Thống Vận Hành Bãi Xe - {message}";
        }
    }

    // ========== MODELS ==========
    public class RecognitionResult
    {
        public bool Success { get; set; }
        public string LicensePlate { get; set; }
        public float Confidence { get; set; }
        public string ImagePath { get; set; }
        public DateTime Timestamp { get; set; }
    }

    public class ApiRecognitionResponse
    {
        [JsonProperty("success")]
        public bool Success { get; set; }

        [JsonProperty("license_plate")]
        public string LicensePlate { get; set; }

        [JsonProperty("confidence")]
        public float Confidence { get; set; }

        [JsonProperty("image_path")]
        public string ImagePath { get; set; }

        [JsonProperty("error")]
        public string Error { get; set; }
    }

    public class VehicleEntry
    {
        public string LicensePlate { get; set; }   // PlateID
        public string DisplayPlate { get; set; }
        public string VehicleType { get; set; }
        public string Slot { get; set; }
        public DateTime EntryTime { get; set; }
    }

    public class CustomerInfo
    {
        public string Name { get; set; }
        public string Phone { get; set; }
        public DateTime ExpiryDate { get; set; }
        public bool IsMonthlyCustomer { get; set; }
    }
}