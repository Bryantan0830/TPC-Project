using System;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace ParallelKMeansMRI
{
    class Program
    {
        // 替换为你电脑上的实际路径
        static readonly string CANCER_FOLDER = @"C:\YourPath\Cancer\";
        static readonly string OUTPUT_FOLDER = @"C:\YourPath\Output\";

        static void Main(string[] args)
        {
            Console.WriteLine("=== Parallel K-Means MRI Segmentation (Color Alert) ===");

            // 1. 获取测试图片的完整路径，并动态生成输出名称
            string testImagePath = CANCER_FOLDER + "Cancer (1).jpg";
            string originalFileName = Path.GetFileName(testImagePath); // 提取文件名
            string savePath = OUTPUT_FOLDER + "segmented_" + originalFileName; // 组合新名字

            // 提取图像像素到一维数组 (无预处理)
            byte[] pixels = LoadImageToByteArray(testImagePath, out int width, out int height);
            Console.WriteLine($"\nProcessing: {originalFileName}");
            Console.WriteLine($"Total Pixels: {pixels.Length}");

            // K=4 (通常分为：背景、灰质、白质、高亮肿瘤)
            int k = 4;
            int iterations = 10;

            // 2. 运行顶级优化并行测试
            Console.WriteLine("\n[Running Optimized Parallel K-Means...]");
            Stopwatch sw = Stopwatch.StartNew();

            // 现在的函数返回的是 每个像素的分类结果 和 最终的中心点亮度
            var (assignments, centroids) = ParallelKMeansOptimized(pixels, k, iterations);

            sw.Stop();
            Console.WriteLine($"Parallel Time: {sw.ElapsedMilliseconds} ms");

            // 3. 将结果渲染为彩色图并保存
            SaveColorSegmentedImage(assignments, centroids, width, height, savePath);
            Console.WriteLine($"\n>>> Success! Saved to: {savePath}");
            Console.ReadLine();
        }

        // --- 核心算法：使用 ThreadLocal 优化的并行 K-Means ---
        // (注意：返回值改成了元组 Tuple，方便我们后续做彩色渲染)
        static (int[] assignments, int[] centroids) ParallelKMeansOptimized(byte[] pixels, int k, int maxIterations)
        {
            int[] centroids = InitializeCentroids(k);
            int[] pixelAssignments = new int[pixels.Length];
            object globalLock = new object();

            for (int iter = 0; iter < maxIterations; iter++)
            {
                long[] globalSums = new long[k];
                int[] globalCounts = new int[k];

                Parallel.For(0, pixels.Length,
                    () => new { localSums = new long[k], localCounts = new int[k] },
                    (i, loopState, localState) =>
                    {
                        int closestCentroid = GetClosestCentroid(pixels[i], centroids);
                        pixelAssignments[i] = closestCentroid;

                        localState.localSums[closestCentroid] += pixels[i];
                        localState.localCounts[closestCentroid]++;
                        return localState;
                    },
                    (localState) =>
                    {
                        lock (globalLock)
                        {
                            for (int j = 0; j < k; j++)
                            {
                                globalSums[j] += localState.localSums[j];
                                globalCounts[j] += localState.localCounts[j];
                            }
                        }
                    }
                );

                for (int j = 0; j < k; j++)
                {
                    if (globalCounts[j] > 0)
                        centroids[j] = (int)(globalSums[j] / globalCounts[j]);
                }
            }

            return (pixelAssignments, centroids);
        }

        // --- 高分视觉渲染机制：肿瘤变红 ---
        static void SaveColorSegmentedImage(int[] assignments, int[] centroids, int width, int height, string savePath)
        {
            // 找查亮度最高的聚类中心 (在 MRI 中，最亮的部分通常是肿瘤或异常积液)
            int maxCentroidValue = centroids.Max();
            int tumorClusterIndex = Array.IndexOf(centroids, maxCentroidValue);

            // 使用 Format24bppRgb 开启彩色输出模式
            using (Bitmap bmp = new Bitmap(width, height, PixelFormat.Format24bppRgb))
            {
                Rectangle rect = new Rectangle(0, 0, width, height);
                BitmapData bmpData = bmp.LockBits(rect, ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);

                int bytes = Math.Abs(bmpData.Stride) * bmp.Height;
                byte[] rgbValues = new byte[bytes];

                // 遍历每一个像素
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int pixelIndex = y * width + x;
                        int cluster = assignments[pixelIndex];

                        // 在 24bpp 图像中，每 3 个字节代表一个像素，顺序是 B-G-R (蓝-绿-红)
                        int byteIndex = y * bmpData.Stride + x * 3;

                        if (cluster == tumorClusterIndex)
                        {
                            // 如果是肿瘤所在的类：强行渲染为纯红色
                            rgbValues[byteIndex] = 0;       // Blue
                            rgbValues[byteIndex + 1] = 0;   // Green
                            rgbValues[byteIndex + 2] = 255; // Red
                        }
                        else
                        {
                            // 如果是正常脑组织：保持算法算出的灰色
                            byte gray = (byte)centroids[cluster];
                            rgbValues[byteIndex] = gray;     // Blue
                            rgbValues[byteIndex + 1] = gray; // Green
                            rgbValues[byteIndex + 2] = gray; // Red
                        }
                    }
                }

                Marshal.Copy(rgbValues, 0, bmpData.Scan0, bytes);
                bmp.UnlockBits(bmpData);
                bmp.Save(savePath, ImageFormat.Jpeg);
            }
        }

        // --- 辅助数学与读取方法 ---
        static int GetClosestCentroid(byte pixelIntensity, int[] centroids)
        {
            int minDistance = int.MaxValue;
            int closestIndex = 0;
            for (int i = 0; i < centroids.Length; i++)
            {
                int distance = Math.Abs(pixelIntensity - centroids[i]);
                if (distance < minDistance)
                {
                    minDistance = distance;
                    closestIndex = i;
                }
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

        static byte[] LoadImageToByteArray(string path, out int width, out int height)
        {
            using (Bitmap bmp = new Bitmap(path))
            {
                width = bmp.Width; height = bmp.Height;
                Rectangle rect = new Rectangle(0, 0, width, height);
                // 读取时依然使用 8bpp 灰度模式以追求极限性能
                BitmapData bmpData = bmp.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format8bppIndexed);

                int bytes = Math.Abs(bmpData.Stride) * bmp.Height;
                byte[] grayValues = new byte[bytes];
                Marshal.Copy(bmpData.Scan0, grayValues, 0, bytes);
                bmp.UnlockBits(bmpData);
                return grayValues;
            }
        }
    }
}
