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
using System.Collections.Generic;
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

        // 【核心修改 1】：加入了 outputFolder 参数
        public static BenchmarkSummary RunBenchmark(string testImagePath, Context context, Device selectedDevice, string outputFolder)
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
                finalResult = SequentialKMeans(pixels, k, iterations);
            }
            summary.FinalImagePixels = finalResult;

            // 【核心修改 2】：使用用户自定义的 outputFolder 进行图片保存
            if (!Directory.Exists(outputFolder)) Directory.CreateDirectory(outputFolder);
            string originalFileName = Path.GetFileNameWithoutExtension(testImagePath);
            string safeDeviceName = selectedDevice != null ? new string(selectedDevice.Name.Where(c => char.IsLetterOrDigit(c)).ToArray()) : "CPU";
            string savePath = Path.Combine(outputFolder, "segmented_" + originalFileName + "_" + safeDeviceName + ".jpg");

            SaveColorHighlightedImage(finalResult, width, height, savePath);

            return summary;
        }

        // ================= ALGORITHM IMPLEMENTATIONS =================

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

        static byte[] ParallelKMeansILGPU(Accelerator accelerator, byte[] pixels, int k, int maxIterations)
        {
            int[] centroids = InitializeCentroids(k);
            int[] assignments = new int[pixels.Length];

            var kernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView1D<byte, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>, ArrayView1D<int, Stride1D.Dense>>(KMeansKernel);

            using var d_pixels = accelerator.Allocate1D<byte>(pixels.Length);
            using var d_centroids = accelerator.Allocate1D<int>(k);
            using var d_assignments = accelerator.Allocate1D<int>(pixels.Length);

            d_pixels.CopyFromCPU(pixels);

            for (int iter = 0; iter < maxIterations; iter++)
            {
                d_centroids.CopyFromCPU(centroids);
                kernel((int)d_pixels.Length, d_pixels.View, d_centroids.View, d_assignments.View);
                accelerator.Synchronize();
                d_assignments.CopyToCPU(assignments);

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
            using (Bitmap original = new Bitmap(path))
            {
                width = original.Width;
                height = original.Height;
                byte[] grayValues = new byte[width * height];

                using (Bitmap bmp32 = new Bitmap(width, height, PixelFormat.Format32bppArgb))
                {
                    using (Graphics g = Graphics.FromImage(bmp32))
                    {
                        g.DrawImage(original, 0, 0, width, height);
                    }

                    BitmapData bmpData = bmp32.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
                    int stride = bmpData.Stride;
                    IntPtr ptr = bmpData.Scan0;
                    int bytes = Math.Abs(stride) * height;
                    byte[] argbValues = new byte[bytes];
                    Marshal.Copy(ptr, argbValues, 0, bytes);
                    bmp32.UnlockBits(bmpData);

                    for (int y = 0; y < height; y++)
                    {
                        for (int x = 0; x < width; x++)
                        {
                            int i = (y * stride) + x * 4;
                            byte b = argbValues[i];
                            byte g = argbValues[i + 1];
                            byte r = argbValues[i + 2];
                            byte gray = (byte)(0.299 * r + 0.587 * g + 0.114 * b);
                            grayValues[y * width + x] = gray;
                        }
                    }
                }
                return grayValues;
            }
        }

        // ================= 智能目标检测(Bounding Box)渲染引擎 V3.0 =================
        public static void SaveColorHighlightedImage(byte[] pixels, int width, int height, string path)
        {
            int headMinX = width, headMaxX = 0, headMinY = height, headMaxY = 0;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    if (pixels[y * width + x] > 20)
                    {
                        if (x < headMinX) headMinX = x;
                        if (x > headMaxX) headMaxX = x;
                        if (y < headMinY) headMinY = y;
                        if (y > headMaxY) headMaxY = y;
                    }
                }
            }
            if (headMaxX <= headMinX) { headMinX = 0; headMaxX = width; headMinY = 0; headMaxY = height; }

            int headArea = (headMaxX - headMinX) * (headMaxY - headMinY);
            double centerX = (headMinX + headMaxX) / 2.0;
            double centerY = (headMinY + headMaxY) / 2.0;

            double rx2 = Math.Pow((headMaxX - headMinX) * 0.38, 2);
            double ry2 = Math.Pow((headMaxY - headMinY) * 0.40, 2);

            var uniqueClusters = pixels.Distinct().OrderByDescending(v => v).ToList();

            bool isTumorFound = false;
            byte targetIntensity = 0;
            int bestMinX = width, bestMaxX = 0, bestMinY = height, bestMaxY = 0;
            bool[] visited = new bool[width * height];

            foreach (byte intensity in uniqueClusters)
            {
                if (intensity < 120) continue;

                int largestArea = 0;
                int currentBestMinX = width, currentBestMaxX = 0, currentBestMinY = height, currentBestMaxY = 0;
                Array.Clear(visited, 0, visited.Length);

                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int idx = y * width + x;

                        if (pixels[idx] == intensity && !visited[idx])
                        {
                            double dx = x - centerX;
                            double dy = y - centerY;
                            if ((dx * dx) / rx2 + (dy * dy) / ry2 > 1.0)
                            {
                                visited[idx] = true;
                                continue;
                            }

                            int area = 0;
                            int minX = x, maxX = x, minY = y, maxY = y;
                            Queue<int> q = new Queue<int>();
                            q.Enqueue(idx);
                            visited[idx] = true;

                            while (q.Count > 0)
                            {
                                int curr = q.Dequeue();
                                area++;
                                int cx = curr % width;
                                int cy = curr / width;

                                if (cx < minX) minX = cx;
                                if (cx > maxX) maxX = cx;
                                if (cy < minY) minY = cy;
                                if (cy > maxY) maxY = cy;

                                int[] dx_arr = { -1, 1, 0, 0 };
                                int[] dy_arr = { 0, 0, -1, 1 };
                                for (int i = 0; i < 4; i++)
                                {
                                    int nx = cx + dx_arr[i];
                                    int ny = cy + dy_arr[i];
                                    if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                                    {
                                        int nIdx = ny * width + nx;
                                        if (pixels[nIdx] == intensity && !visited[nIdx])
                                        {
                                            double ndx = nx - centerX;
                                            double ndy = ny - centerY;
                                            if ((ndx * ndx) / rx2 + (ndy * ndy) / ry2 > 1.0)
                                            {
                                                visited[nIdx] = true;
                                                continue;
                                            }

                                            visited[nIdx] = true;
                                            q.Enqueue(nIdx);
                                        }
                                    }
                                }
                            }

                            if (area > largestArea)
                            {
                                largestArea = area;
                                currentBestMinX = minX; currentBestMaxX = maxX;
                                currentBestMinY = minY; currentBestMaxY = maxY;
                            }
                        }
                    }
                }

                if (largestArea > (headArea * 0.001) && largestArea < (headArea * 0.15))
                {
                    isTumorFound = true;
                    targetIntensity = intensity;
                    bestMinX = currentBestMinX;
                    bestMaxX = currentBestMaxX;
                    bestMinY = currentBestMinY;
                    bestMaxY = currentBestMaxY;
                    break;
                }
            }

            using (Bitmap bmp = new Bitmap(width, height, PixelFormat.Format24bppRgb))
            {
                BitmapData bmpData = bmp.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);
                int bytes = Math.Abs(bmpData.Stride) * bmp.Height;
                byte[] rgbValues = new byte[bytes];

                int padding = 6;
                int thickness = 3;

                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int pixelIndex = y * width + x;
                        int byteIndex = y * bmpData.Stride + x * 3;
                        byte currentIntensity = pixels[pixelIndex];

                        rgbValues[byteIndex] = currentIntensity;
                        rgbValues[byteIndex + 1] = currentIntensity;
                        rgbValues[byteIndex + 2] = currentIntensity;

                        if (isTumorFound)
                        {
                            bool isBorderY = (y >= bestMinY - padding - thickness && y <= bestMinY - padding) ||
                                             (y >= bestMaxY + padding && y <= bestMaxY + padding + thickness);
                            bool isBorderX = (x >= bestMinX - padding - thickness && x <= bestMinX - padding) ||
                                             (x >= bestMaxX + padding && x <= bestMaxX + padding + thickness);

                            bool withinBoxX = x >= bestMinX - padding - thickness && x <= bestMaxX + padding + thickness;
                            bool withinBoxY = y >= bestMinY - padding - thickness && y <= bestMaxY + padding + thickness;

                            if ((isBorderY && withinBoxX) || (isBorderX && withinBoxY))
                            {
                                rgbValues[byteIndex] = 0;
                                rgbValues[byteIndex + 1] = 0;
                                rgbValues[byteIndex + 2] = 255;
                            }
                            else if (currentIntensity == targetIntensity && x >= bestMinX && x <= bestMaxX && y >= bestMinY && y <= bestMaxY)
                            {
                                rgbValues[byteIndex] = 70;
                                rgbValues[byteIndex + 1] = 70;
                                rgbValues[byteIndex + 2] = 255;
                            }
                        }
                    }
                }

                Marshal.Copy(rgbValues, 0, bmpData.Scan0, bytes);
                bmp.UnlockBits(bmpData);
                bmp.Save(path, ImageFormat.Jpeg);
            }
        }
    }
}
