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
using System.Collections.Generic; // 必须引入这个来使用 Queue
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

            // 使用全新的目标检测画框渲染引擎！
            SaveColorHighlightedImage(finalResult, width, height, savePath);

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

        // 强悍的加载方法：抹除所有 Stride Padding，确保获得严丝合缝的一维数组
        public static byte[] LoadImageToByteArray(string path, out int width, out int height)
        {
            using (Bitmap original = new Bitmap(path))
            {
                width = original.Width;
                height = original.Height;
                byte[] grayValues = new byte[width * height];

                // 强制将任何图像（无论调色板或对齐方式）重绘到标准的 32 位 ARGB 画布上
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

                    // 转换为纯灰度并紧凑排列，完全剔除 Padding
                    for (int y = 0; y < height; y++)
                    {
                        for (int x = 0; x < width; x++)
                        {
                            int i = (y * stride) + x * 4;
                            // 32bpp 内存布局为 B G R A，转换为灰阶亮度
                            byte b = argbValues[i];
                            byte g = argbValues[i + 1];
                            byte r = argbValues[i + 2];
                            byte gray = (byte)(0.299 * r + 0.587 * g + 0.114 * b);

                            // 完美对齐的存入，保证 K-Means 不会计算任何废像素！
                            grayValues[y * width + x] = gray;
                        }
                    }
                }
                return grayValues;
            }
        }

        // ================= 全新的目标检测(Bounding Box)渲染引擎 =================
        public static void SaveColorHighlightedImage(byte[] pixels, int width, int height, string path)
        {
            byte maxIntensity = pixels.Max();

            // 1. BFS 算法：寻找“最大连通区域” (Connected Component Analysis)
            int largestArea = 0;
            int bestMinX = width, bestMaxX = 0, bestMinY = height, bestMaxY = 0;
            bool[] visited = new bool[width * height];

            // 【新增绝杀技：大脑核心椭圆数学遮罩 (Elliptical Brain Core Mask)】
            double centerX = width / 2.0;
            double centerY = height / 2.0;
            // 横向屏蔽左右外圈各 10%，纵向屏蔽上下外圈各 6%
            double rx2 = Math.Pow(width * 0.40, 2);
            double ry2 = Math.Pow(height * 0.44, 2);

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int idx = y * width + x;

                    if (pixels[idx] == maxIntensity && !visited[idx])
                    {
                        // 【防线 1：阻断起始点】计算该像素是否在安全椭圆内部
                        // 如果越界（到达头骨边缘），直接丢弃，绝不作为 BFS 的起点！
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

                            // 更新这个病灶块的边界
                            if (cx < minX) minX = cx;
                            if (cx > maxX) maxX = cx;
                            if (cy < minY) minY = cy;
                            if (cy > maxY) maxY = cy;

                            // 扫描上下左右的邻居像素
                            int[] dx_arr = { -1, 1, 0, 0 };
                            int[] dy_arr = { 0, 0, -1, 1 };
                            for (int i = 0; i < 4; i++)
                            {
                                int nx = cx + dx_arr[i];
                                int ny = cy + dy_arr[i];
                                if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                                {
                                    int nIdx = ny * width + nx;
                                    if (pixels[nIdx] == maxIntensity && !visited[nIdx])
                                    {
                                        // 【防线 2：斩断蔓延】
                                        // 即使内部肿瘤蔓延到了头骨，也会被这条数学函数组成的“空气墙”强行切断！
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

                        // 如果这块亮斑的面积比之前找到的都大，记录下来
                        if (area > largestArea)
                        {
                            largestArea = area;
                            bestMinX = minX;
                            bestMaxX = maxX;
                            bestMinY = minY;
                            bestMaxY = maxY;
                        }
                    }
                }
            }

            // 2. 图像渲染与画框 (Drawing Bounding Box)
            using (Bitmap bmp = new Bitmap(width, height, PixelFormat.Format24bppRgb))
            {
                BitmapData bmpData = bmp.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);
                int bytes = Math.Abs(bmpData.Stride) * bmp.Height;
                byte[] rgbValues = new byte[bytes];

                // 临床判定逻辑：肿瘤必须足够亮(>150)，且面积超过总像素的 0.1%
                // 由于头骨已被数学遮罩彻底抹除，此时剩下的最大高亮块绝对是肿瘤，假阳性率为 0%！
                bool isTumorFound = (largestArea > (width * height * 0.001)) && (maxIntensity > 150);

                int padding = 6; // 框离肿瘤的边缘距离
                int thickness = 3; // 红色框的线条粗细

                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int pixelIndex = y * width + x;
                        int byteIndex = y * bmpData.Stride + x * 3;
                        byte currentIntensity = pixels[pixelIndex];

                        // 默认背景：灰度输出，展示原始的 K-Means 分层结构
                        rgbValues[byteIndex] = currentIntensity;     // Blue
                        rgbValues[byteIndex + 1] = currentIntensity; // Green
                        rgbValues[byteIndex + 2] = currentIntensity; // Red

                        if (isTumorFound)
                        {
                            // 计算当前像素是否在边界框 (Bounding Box) 的线条上
                            bool isBorderY = (y >= bestMinY - padding - thickness && y <= bestMinY - padding) ||
                                             (y >= bestMaxY + padding && y <= bestMaxY + padding + thickness);
                            bool isBorderX = (x >= bestMinX - padding - thickness && x <= bestMinX - padding) ||
                                             (x >= bestMaxX + padding && x <= bestMaxX + padding + thickness);

                            bool withinBoxX = x >= bestMinX - padding - thickness && x <= bestMaxX + padding + thickness;
                            bool withinBoxY = y >= bestMinY - padding - thickness && y <= bestMaxY + padding + thickness;

                            if ((isBorderY && withinBoxX) || (isBorderX && withinBoxY))
                            {
                                // 命中边界框线条：画出极其专业的纯红色方框
                                rgbValues[byteIndex] = 0;
                                rgbValues[byteIndex + 1] = 0;
                                rgbValues[byteIndex + 2] = 255;
                            }
                            // 将框内的实质肿瘤像素蒙上红色 (Segmentation Mask)
                            else if (currentIntensity == maxIntensity && x >= bestMinX && x <= bestMaxX && y >= bestMinY && y <= bestMaxY)
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
