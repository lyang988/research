using System;
using System.Collections.Generic;

namespace FractalTerrain {
    class Point {
        public double x;
        public double y;
    }

    class Program {
        static void Main(string[] args) {
            int iterations = 8;
            double roughness = 0.5;
            double displacementFactor = 1.0;
            Random generator = new Random();
            double[] ys = { 0.0, 0.0 };

            for (int i = 0; i < iterations; i++) {
                ys = Split(ys, displacementFactor, generator);
                displacementFactor *= roughness;
            }

            Point[] points = ComposeCoordinatePairs(ys);
            foreach (Point p in points) {
                Console.WriteLine("{0:0.000} {1:0.000}", p.x, p.y);
            }
        }

        static double[] Split(double[] ys, double displacementFactor, Random generator) {
            double[] newYs = new double[2 * ys.Length - 1];

            for (int i = 0; i < ys.Length - 1; i++) {
                double midpoint = (ys[i] + ys[i + 1]) / 2.0;
                double displacement = generator.NextDouble() * displacementFactor;
                newYs[2 * i] = ys[i];
                newYs[2 * i + 1] = midpoint + displacement;
            }

            return newYs;
        }

        static Point[] ComposeCoordinatePairs(double[] ys) {
            double increment = 1.0 / (ys.Length - 1);
            Point[] points = new Point[ys.Length];

            for (int i = 0; i < ys.Length; i++) {
                points[i] = new Point();
                points[i].x = increment * i;
                points[i].y = ys[i];
            }

            return points;
        }
    }
}