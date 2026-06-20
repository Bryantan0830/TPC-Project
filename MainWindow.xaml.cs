using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Imaging;
using ILGPU;
using ILGPU.Runtime;
using System.Collections.Generic;

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

        private async void RunButton_Click(object sender, RoutedEventArgs e)
        {
            RunButton.IsEnabled = false;
            ResultsDataGrid.ItemsSource = null;
            StatusText.Text = "Running benchmark... Please wait.";

            string imagePath = ImageComboBox.SelectedIndex == 0
                ? Path.Combine(KMeansEngine.CANCER_FOLDER, "Cancer (1).jpg")
                : Path.Combine(KMeansEngine.NOT_CANCER_FOLDER, "Not Cancer  (1).jpg");

            Device selectedDevice = null;
            if (HardwareComboBox.SelectedIndex > 0)
            {
                selectedDevice = _devices[HardwareComboBox.SelectedIndex - 1];
            }

            try
            {
                // Show original image
                OriginalImage.Source = new BitmapImage(new Uri(Path.GetFullPath(imagePath)));

                var result = await Task.Run(() => KMeansEngine.RunBenchmark(imagePath, _context, selectedDevice));

                ResultsDataGrid.ItemsSource = result.Results;
                
                string safeDeviceName = selectedDevice != null ? new string(selectedDevice.Name.Where(c => char.IsLetterOrDigit(c)).ToArray()) : "CPU";
                string savePath = Path.Combine(KMeansEngine.OUTPUT_FOLDER, "segmented_" + Path.GetFileNameWithoutExtension(imagePath) + "_" + safeDeviceName + ".jpg");

                // Show segmented image
                BitmapImage bitmap = new BitmapImage();
                bitmap.BeginInit();
                bitmap.CacheOption = BitmapCacheOption.OnLoad;
                bitmap.UriSource = new Uri(Path.GetFullPath(savePath));
                bitmap.EndInit();
                SegmentedImage.Source = bitmap;

                StatusText.Text = "Benchmark completed successfully.";
            }
            catch (Exception ex)
            {
                StatusText.Text = $"Error: {ex.Message}";
            }
            finally
            {
                RunButton.IsEnabled = true;
            }
        }
    }
}
