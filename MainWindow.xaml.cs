using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Imaging;
using ILGPU;
using ILGPU.Runtime;
using System.Collections.Generic;
using Microsoft.Win32;
using System.Windows.Media;

namespace ParallelKMeansMRI
{
    public partial class MainWindow : Window
    {
        private Context _context;
        private List<Device> _devices;

        public MainWindow()
        {
            InitializeComponent();
            Loaded += MainWindow_Loaded;
        }

        private void MainWindow_Loaded(object sender, RoutedEventArgs e)
        {
            try
            {
                StatusText.Text = "Initializing ILGPU Context...";
                _context = Context.CreateDefault();
                _devices = _context.Devices.ToList();

                HardwareComboBox.Items.Add("CPU Only (Skip ILGPU)");
                foreach (var device in _devices)
                {
                    HardwareComboBox.Items.Add($"{device.Name} ({device.AcceleratorType})");
                }
                HardwareComboBox.SelectedIndex = 1 < HardwareComboBox.Items.Count ? 1 : 0;

                ImagePathTextBox.Text = Path.Combine(KMeansEngine.CANCER_FOLDER, "Cancer (1).jpg");

                // 初始化输出路径
                if (OutputPathTextBox != null)
                {
                    OutputPathTextBox.Text = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Data", "Output");
                }

                StatusText.Text = "Ready";
            }
            catch (Exception ex)
            {
                StatusText.Text = $"Failed to initialize ILGPU: {ex.Message}";
            }
        }

        protected override void OnClosed(EventArgs e)
        {
            _context?.Dispose();
            base.OnClosed(e);
        }

        // 恢复为单张图片选择器 (OpenFileDialog)
        private void BrowseButton_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Title = "Select MRI Image";
            openFileDialog.Filter = "Image Files (*.jpg;*.png;*.tif;*.bmp)|*.jpg;*.png;*.tif;*.bmp|All Files (*.*)|*.*";
            openFileDialog.InitialDirectory = Path.GetFullPath("Brain Tumor Data Set");

            if (openFileDialog.ShowDialog() == true)
            {
                ImagePathTextBox.Text = openFileDialog.FileName;
            }
        }

        // 输出文件夹选择器
        private void BrowseOutputButton_Click(object sender, RoutedEventArgs e)
        {
            OpenFolderDialog openFolderDialog = new OpenFolderDialog();
            openFolderDialog.Title = "Select Destination Folder for Segmented MRI";

            if (openFolderDialog.ShowDialog() == true)
            {
                if (OutputPathTextBox != null)
                {
                    OutputPathTextBox.Text = openFolderDialog.FolderName;
                }
            }
        }

        private async void RunButton_Click(object sender, RoutedEventArgs e)
        {
            RunButton.IsEnabled = false;
            ResultsDataGrid.ItemsSource = null;
            StatusText.Foreground = Brushes.Black;
            StatusText.Text = "Running benchmark... Please wait.";

            string imagePath = ImagePathTextBox.Text;

            // 获取用户填写的输出路径
            string outputPath = (OutputPathTextBox != null && !string.IsNullOrWhiteSpace(OutputPathTextBox.Text))
                                ? OutputPathTextBox.Text
                                : Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Data", "Output");

            if (string.IsNullOrWhiteSpace(imagePath) || !File.Exists(imagePath))
            {
                StatusText.Foreground = Brushes.Red;
                StatusText.Text = "Error: Please select a valid single image file first.";
                RunButton.IsEnabled = true;
                return;
            }

            Device selectedDevice = null;
            if (HardwareComboBox.SelectedIndex > 0)
            {
                selectedDevice = _devices[HardwareComboBox.SelectedIndex - 1];
            }

            try
            {
                // 展示原图
                BitmapImage origBitmap = new BitmapImage();
                origBitmap.BeginInit();
                origBitmap.CacheOption = BitmapCacheOption.OnLoad;
                origBitmap.UriSource = new Uri(Path.GetFullPath(imagePath));
                origBitmap.EndInit();
                OriginalImage.Source = origBitmap;

                // 后台执行算法 Benchmark，传入自定义的 outputPath
                var result = await Task.Run(() => KMeansEngine.RunBenchmark(imagePath, _context, selectedDevice, outputPath));

                // 将跑分数据绑定给表格
                ResultsDataGrid.ItemsSource = result.Results;

                string safeDeviceName = selectedDevice != null ? new string(selectedDevice.Name.Where(c => char.IsLetterOrDigit(c)).ToArray()) : "CPU";
                string savePath = Path.Combine(outputPath, "segmented_" + Path.GetFileNameWithoutExtension(imagePath) + "_" + safeDeviceName + ".jpg");

                // 展示最终带有红框的预测图
                BitmapImage bitmap = new BitmapImage();
                bitmap.BeginInit();
                bitmap.CacheOption = BitmapCacheOption.OnLoad;
                bitmap.UriSource = new Uri(Path.GetFullPath(savePath));
                bitmap.EndInit();
                SegmentedImage.Source = bitmap;

                // AI 智能调度：找出最快算法并播报
                var optimalAlgorithm = result.Results
                    .Where(r => r.TimeMs > 0)
                    .OrderBy(r => r.TimeMs)
                    .First();

                StatusText.Foreground = Brushes.DarkGreen;
                StatusText.Text = $"⚡ AI Smart Dispatcher auto-selected optimal architecture [{optimalAlgorithm.AlgorithmName}]. Inference time: {optimalAlgorithm.TimeMs} ms.";
            }
            catch (Exception ex)
            {
                StatusText.Foreground = Brushes.Red;
                StatusText.Text = $"Error: {ex.Message}";
            }
            finally
            {
                RunButton.IsEnabled = true;
            }
        }
    }
}
