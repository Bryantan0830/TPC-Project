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

        private async void RunButton_Click(object sender, RoutedEventArgs e)
        {
            RunButton.IsEnabled = false;
            ResultsDataGrid.ItemsSource = null;
            StatusText.Text = "Running benchmark... Please wait.";

            string imagePath = ImagePathTextBox.Text;

            if (string.IsNullOrWhiteSpace(imagePath) || !File.Exists(imagePath))
            {
                StatusText.Text = "Error: Please select a valid image file first.";
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
