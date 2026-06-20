using System;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;

namespace ParallelKMeansMRI
{
    public class BenchmarkResult
    {
        public string AlgorithmName { get; set; }
        public long TimeMs { get; set; }
        public double Speedup { get; set; }
    }

    public class BenchmarkSummary
    {
        public List<BenchmarkResult> Results { get; set; } = new List<BenchmarkResult>();
        public byte[] FinalImagePixels { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
    }

    public class KMeansEngine
    {
        public static readonly string CANCER_FOLDER = @"Brain Tumor Data Set\Brain Tumor\";
        public static readonly string NOT_CANCER_FOLDER = @"Brain Tumor Data Set\Healthy\";
        public static readonly string OUTPUT_FOLDER = @"Data\Output\";

        public static BenchmarkSummary RunBenchmark(string testImagePath, Context context, Device selectedDevice)
        {
            if (!File.Exists(testImagePath))
                throw new FileNotFoundException($"Cannot find: {testImagePath}");

            var summary = new BenchmarkSummary();

            byte[] pixels = LoadImageToByteArray(testImagePath, out int width, out int height);
            summary.Width = width;
            summary.Height = height;

            int k = 4;
            int iterations = 10;
            byte[] finalResult = null;

            // 0. Sequential Baseline
            var sw = Stopwatch.StartNew();
            SequentialKMeans(pixels, k, iterations);
            sw.Stop();
            long seqTime = sw.ElapsedMilliseconds;
            summary.Results.Add(new BenchmarkResult { AlgorithmName = "Sequential Baseline", TimeMs = seqTime, Speedup = 1.0 });

            // 1. Basic Parallel (Global Lock)
            sw.Restart();
            ParallelKMeansBasic(pixels, k, iterations);
            sw.Stop();
            summary.Results.Add(new BenchmarkResult { AlgorithmName = "Basic Parallel (Locks)", TimeMs = sw.ElapsedMilliseconds, Speedup = (double)seqTime / Math.Max(1, sw.ElapsedMilliseconds) });

            // 2. ThreadLocal Optimized Parallel
            sw.Restart();
            ParallelKMeansThreadLocal(pixels, k, iterations);
            sw.Stop();
            summary.Results.Add(new BenchmarkResult { AlgorithmName = "ThreadLocal Optimized", TimeMs = sw.ElapsedMilliseconds, Speedup = (double)seqTime / Math.Max(1, sw.ElapsedMilliseconds) });

            // 3. Data Partitioning Parallel
            sw.Restart();
            ParallelKMeansPartitioned(pixels, k, iterations);
            sw.Stop();
            summary.Results.Add(new BenchmarkResult { AlgorithmName = "Data Partitioning", TimeMs = sw.ElapsedMilliseconds, Speedup = (double)seqTime / Math.Max(1, sw.ElapsedMilliseconds) });

            // 4. Task-Based Asynchronous Parallel
            sw.Restart();
            ParallelKMeansTaskBased(pixels, k, iterations);
            sw.Stop();
            summary.Results.Add(new BenchmarkResult { AlgorithmName = "Task-Based Asynchronous", TimeMs = sw.ElapsedMilliseconds, Speedup = (double)seqTime / Math.Max(1, sw.ElapsedMilliseconds) });

            // 5. Test the selected hardware accelerator
            if (selectedDevice != null)
            {
                try
                {
                    using var accelerator = selectedDevice.CreateAccelerator(context);
                    sw.Restart();
                    finalResult = ParallelKMeansILGPU(accelerator, pixels, k, iterations);
                    sw.Stop();
                    summary.Results.Add(new BenchmarkResult { AlgorithmName = $"ILGPU: {selectedDevice.Name}", TimeMs = sw.ElapsedMilliseconds, Speedup = (double)seqTime / Math.Max(1, sw.ElapsedMilliseconds) });
                }
                catch (Exception ex)
                {
                    summary.Results.Add(new BenchmarkResult { AlgorithmName = $"ILGPU: FAILED ({ex.Message})", TimeMs = -1, Speedup = 0 });
                }
            }

            if (finalResult == null)
            {
                finalResult = SequentialKMeans(pixels, k, iterations); // Fallback
            }
            summary.FinalImagePixels = finalResult;

            if (!Directory.Exists(OUTPUT_FOLDER)) Directory.CreateDirectory(OUTPUT_FOLDER);
            string originalFileName = Path.GetFileNameWithoutExtension(testImagePath);
            string safeDeviceName = selectedDevice != null ? new string(selectedDevice.Name.Where(c => char.IsLetterOrDigit(c)).ToArray()) : "CPU";
            string savePath = Path.Combine(OUTPUT_FOLDER, "segmented_" + originalFileName + "_" + safeDeviceName + ".jpg");
            SaveByteArrayToImage(finalResult, width, height, savePath);

            return summary;
        }

        // ================= ALGORITHM IMPLEMENTATIONS =================

        // [0] Baseline Sequential
        static byte[] SequentialKMeans(byte[] pixels, int k, int maxIterations)
        {
            int[] centroids = InitializeCentroids(k);
            int[] assignments = new int[pixels.Length];
            for (int iter = 0; iter < maxIterations; iter++)
            {
                long[] sums = new long[k];
                int[] counts = new int[k];
                for (int i = 0; i < pixels.Length; i++)
                {
                    int c = GetClosestCentroid(pixels[i], centroids);
                    assignments[i] = c;
                    sums[c] += pixels[i];
                    counts[c]++;
                }
                UpdateCentroids(centroids, sums, counts, k);
            }
            return ApplyCentroids(pixels, assignments, centroids);
        }

        // [1] Alg 1: Basic Parallel with Global Lock
        static byte[] ParallelKMeansBasic(byte[] pixels, int k, int maxIterations)
        {
            int[] centroids = InitializeCentroids(k);
            int[] assignments = new int[pixels.Length];
            object globalLock = new object();
            for (int iter = 0; iter < maxIterations; iter++)
            {
                long[] sums = new long[k];
                int[] counts = new int[k];
                Parallel.For(0, pixels.Length, i =>
                {
                    int c = GetClosestCentroid(pixels[i], centroids);
                    assignments[i] = c;
                    lock (globalLock)
                    {
                        sums[c] += pixels[i];
                        counts[c]++;
                    }
                });
                UpdateCentroids(centroids, sums, counts, k);
            }
            return ApplyCentroids(pixels, assignments, centroids);
        }

        // [2] Alg 2: ThreadLocal Optimized
        static byte[] ParallelKMeansThreadLocal(byte[] pixels, int k, int maxIterations)
        {
            int[] centroids = InitializeCentroids(k);
            int[] assignments = new int[pixels.Length];
            object globalLock = new object();
            for (int iter = 0; iter < maxIterations; iter++)
            {
                long[] sums = new long[k];
                int[] counts = new int[k];
                Parallel.For(0, pixels.Length,
                    () => new { localSums = new long[k], localCounts = new int[k] },
                    (i, loopState, localState) =>
                    {
                        int c = GetClosestCentroid(pixels[i], centroids);
                        assignments[i] = c;
                        localState.localSums[c] += pixels[i];
                        localState.localCounts[c]++;
                        return localState;
                    },
                    (localState) =>
                    {
                        lock (globalLock)
                        {
                            for (int j = 0; j < k; j++) { sums[j] += localState.localSums[j]; counts[j] += localState.localCounts[j]; }
                        }
                    }
                );
                UpdateCentroids(centroids, sums, counts, k);
            }
            return ApplyCentroids(pixels, assignments, centroids);
        }

        // [3] Alg 3: Data Partitioning Parallel (Chunks)
        static byte[] ParallelKMeansPartitioned(byte[] pixels, int k, int maxIterations)
        {
            int[] centroids = InitializeCentroids(k);
            int[] assignments = new int[pixels.Length];
            object globalLock = new object();
            for (int iter = 0; iter < maxIterations; iter++)
            {
                long[] sums = new long[k];
                int[] counts = new int[k];
                Parallel.ForEach(Partitioner.Create(0, pixels.Length), range =>
                {
                    long[] localSums = new long[k];
                    int[] localCounts = new int[k];
                    for (int i = range.Item1; i < range.Item2; i++)
                    {
                        int c = GetClosestCentroid(pixels[i], centroids);
                        assignments[i] = c;
                        localSums[c] += pixels[i];
                        localCounts[c]++;
                    }
                    lock (globalLock)
                    {
                        for (int j = 0; j < k; j++) { sums[j] += localSums[j]; counts[j] += localCounts[j]; }
                    }
                });
                UpdateCentroids(centroids, sums, counts, k);
            }
            return ApplyCentroids(pixels, assignments, centroids);
        }

        // [4] Alg 4: Task-Based Asynchronous Parallel
        static byte[] ParallelKMeansTaskBased(byte[] pixels, int k, int maxIterations)
        {
            int[] centroids = InitializeCentroids(k);
            int[] assignments = new int[pixels.Length];
            object globalLock = new object();
            int coreCount = Environment.ProcessorCount;
            
            for (int iter = 0; iter < maxIterations; iter++)
            {
                long[] sums = new long[k];
                int[] counts = new int[k];
                Task[] tasks = new Task[coreCount];
                int chunkSize = (int)Math.Ceiling((double)pixels.Length / coreCount);

                for (int t = 0; t < coreCount; t++)
                {
                    int start = t * chunkSize;
                    int end = Math.Min(start + chunkSize, pixels.Length);
                    tasks[t] = Task.Run(() =>
                    {
                        long[] localSums = new long[k];
                        int[] localCounts = new int[k];
                        for (int i = start; i < end; i++)
                        {
                            int c = GetClosestCentroid(pixels[i], centroids);
                            assignments[i] = c;
                            localSums[c] += pixels[i];
                            localCounts[c]++;
                        }
                        lock (globalLock)
                        {
                            for (int j = 0; j < k; j++) { sums[j] += localSums[j]; counts[j] += localCounts[j]; }
                        }
                    });
                }
                Task.WaitAll(tasks);
                UpdateCentroids(centroids, sums, counts, k);
            }
            return ApplyCentroids(pixels, assignments, centroids);
        }

        // [5] Alg 5: ILGPU Accelerator Parallel
        static byte[] ParallelKMeansILGPU(Accelerator accelerator, byte[] pixels, int k, int maxIterations)
        {
            int[] centroids = InitializeCentroids(k);
            int[] assignments = new int[pixels.Length];
            
            // Load and compile Kernel
            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<byte, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(KMeansKernel);

            // Allocate Memory on Accelerator
            using var d_pixels = accelerator.Allocate1D<byte>(pixels.Length);
            using var d_centroids = accelerator.Allocate1D<int>(k);
            using var d_assignments = accelerator.Allocate1D<int>(pixels.Length);

            // Copy Memory CPU -> GPU
            d_pixels.CopyFromCPU(pixels);

            for (int iter = 0; iter < maxIterations; iter++)
            {
                d_centroids.CopyFromCPU(centroids);
                
                // Execute Kernel on thousands of cores
                kernel((int)d_pixels.Length, d_pixels.View, d_centroids.View, d_assignments.View);
                
                // Wait for GPU to finish
                accelerator.Synchronize();
                
                // Copy Memory GPU -> CPU
                d_assignments.CopyToCPU(assignments);

                // Update Centroids locally (since GPU atomics on global memory for large arrays can be a bottleneck)
                long[] sums = new long[k];
                int[] counts = new int[k];
                for (int i = 0; i < pixels.Length; i++)
                {
                    int c = assignments[i];
                    sums[c] += pixels[i];
                    counts[c]++;
                }
                UpdateCentroids(centroids, sums, counts, k);
            }

            return ApplyCentroids(pixels, assignments, centroids);
        }

        // ILGPU Kernel Code (Runs entirely on Accelerator)
        static void KMeansKernel(
            Index1D index,
            ArrayView1D<byte, Stride1D.Dense> pixels,
            ArrayView1D<int, Stride1D.Dense> centroids,
            ArrayView1D<int, Stride1D.Dense> assignments)
        {
            int k = (int)centroids.Length;
            int minDistance = int.MaxValue;
            int closestIndex = 0;
            byte pixelIntensity = pixels[index];

            for (int i = 0; i < k; i++)
            {
                // Cannot use System.Math inside Kernel easily, write manual Absolute Value
                int diff = pixelIntensity - centroids[i];
                int dist = diff > 0 ? diff : -diff;

                if (dist < minDistance)
                {
                    minDistance = dist;
                    closestIndex = i;
                }
            }
            assignments[index] = closestIndex;
        }

        // ================= HELPERS =================

        static int GetClosestCentroid(byte pixelIntensity, int[] centroids)
        {
            int minDistance = int.MaxValue;
            int closestIndex = 0;
            for (int i = 0; i < centroids.Length; i++)
            {
                int dist = Math.Abs(pixelIntensity - centroids[i]);
                if (dist < minDistance) { minDistance = dist; closestIndex = i; }
            }
            return closestIndex;
        }

        static int[] InitializeCentroids(int k)
        {
            int[] centroids = new int[k];
            int step = 255 / k;
            for (int i = 0; i < k; i++) centroids[i] = (i * step) + (step / 2);
            return centroids;
        }

        static void UpdateCentroids(int[] centroids, long[] sums, int[] counts, int k)
        {
            for (int j = 0; j < k; j++)
                if (counts[j] > 0) centroids[j] = (int)(sums[j] / counts[j]);
        }

        static byte[] ApplyCentroids(byte[] pixels, int[] assignments, int[] centroids)
        {
            byte[] result = new byte[pixels.Length];
            for (int i = 0; i < pixels.Length; i++) result[i] = (byte)centroids[assignments[i]];
            return result;
        }

        public static byte[] LoadImageToByteArray(string path, out int width, out int height)
        {
            using (Bitmap bmp = new Bitmap(path))
            {
                width = bmp.Width; height = bmp.Height;
                BitmapData bmpData = bmp.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadOnly, PixelFormat.Format8bppIndexed);
                int bytes = Math.Abs(bmpData.Stride) * bmp.Height;
                byte[] grayValues = new byte[bytes];
                Marshal.Copy(bmpData.Scan0, grayValues, 0, bytes);
                bmp.UnlockBits(bmpData);
                return grayValues;
            }
        }

        public static void SaveByteArrayToImage(byte[] pixels, int width, int height, string path)
        {
            using (Bitmap bmp = new Bitmap(width, height, PixelFormat.Format8bppIndexed))
            {
                ColorPalette pal = bmp.Palette;
                for (int i = 0; i < 256; i++) pal.Entries[i] = Color.FromArgb(255, i, i, i);
                bmp.Palette = pal;
                BitmapData bmpData = bmp.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, PixelFormat.Format8bppIndexed);
                Marshal.Copy(pixels, 0, bmpData.Scan0, pixels.Length);
                bmp.UnlockBits(bmpData);
                bmp.Save(path, ImageFormat.Jpeg);
            }
        }
    }
}
