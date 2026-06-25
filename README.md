# Parallel K-Means MRI Segmentation

This is a WPF desktop application designed to perform K-Means clustering for Brain Tumor MRI image segmentation. The core focus of this project is to implement and benchmark various parallelization strategies to accelerate the image segmentation process.

## Features

- **MRI Image Segmentation:** Select and load brain tumor MRI images.
- **Multiple Algorithms:** Benchmarks different implementations of K-Means clustering:
  - Sequential Baseline
  - Basic Parallel (Global Lock)
  - Thread-Local Optimized Parallel
  - Data Partitioning Parallel
  - Task-Based Asynchronous Parallel
  - GPU-Accelerated Parallel (using ILGPU)
- **Interactive UI:** A WPF-based graphical user interface to choose images, select the target hardware accelerator (CPU or specific GPU), and view the original and segmented results side-by-side.
- **Performance Benchmarking:** Displays execution time and speedup metrics for each algorithmic approach.

## Technologies Used

- **C#** and **WPF** (Windows Presentation Foundation)
- **.NET 10.0**
- **ILGPU** (for GPU acceleration)
- **System.Drawing.Common** (for image processing)

## Getting Started

### Prerequisites

- [.NET 10.0 SDK](https://dotnet.microsoft.com/download)
- A compatible GPU (optional, but required for ILGPU acceleration).
- Visual Studio 2022 (recommended) or any compatible C# IDE.

### Installation & Running

1. Clone or download this repository.
2. Open the `TPC-Project.csproj` in your preferred IDE (e.g., Visual Studio).
3. Build the project to restore NuGet packages (like `ILGPU`).
4. Run the application.

### Usage

1. Launch the application.
2. The system will automatically detect available ILGPU contexts and hardware accelerators.
3. Select an accelerator from the dropdown (CPU or your GPU).
4. Click **Browse** to select an MRI image (supported formats: `.jpg`, `.png`, `.tif`, `.bmp`). The application defaults to the `Brain Tumor Data Set` folder.
5. Click **Run Benchmark**. The application will execute all implemented K-Means algorithms, record their execution times, and calculate the speedup compared to the sequential baseline.
6. The segmented image will be displayed and automatically saved to the `Data\Output` folder.

## Project Structure

- `MainWindow.xaml` & `MainWindow.xaml.cs`: The main UI logic and layout.
- `KMeansEngine.cs`: Contains the core logic for the different K-Means clustering algorithms.
- `Brain Tumor Data Set/`: Directory containing sample MRI images.
- `Data/Output/`: Directory where the segmented images are saved.
