﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;

using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using Emgu.CV.Util;
using System.Diagnostics;
using Emgu.CV.OCR;

using System.Timers;

namespace Basket
{
    class Program
    {
        static void Main(string[] args)
        {
            var ai = new BasketAI();
            ai.Start();

            //var ai = new AIModule();
            //ai.Start();
        }
    }

    public static class Extensions
    {
        public static Point ToPoint(this PointF pf) => new Point((int)(Math.Round(pf.X)), (int)(Math.Round(pf.Y)));

        public static PointF ToPointF(this Point p) => new PointF((float)p.X, (float)p.Y);

        public static float Length(this PointF pf) => (float)Math.Sqrt((double)(pf.X * pf.X + pf.Y * pf.Y));

        public static MCvScalar ToScalar(this Color col) => new Bgr(col).MCvScalar;
    }

    public static class Helpers
    {
        public static List<string> DebugInfoList { get; private set; } = new List<string>();

        public static void DebugInfo(string fmt, params object[] args) => DebugInfoList.Add(string.Format(fmt, args));

        public static void AddDebugInfoAndDraw(string window, IInputOutputArray image)
        {
            int startX = 5;
            int startY = 15;
            int deltaY = 20;

            for (int i = 0; i < DebugInfoList.Count; i++)
            {
                CvInvoke.PutText(image, DebugInfoList[i], new Point(startX, (int)(startY + (deltaY * i))), FontFace.HersheySimplex, 0.5, new Bgr(0, 255, 0).MCvScalar, 1);
            }

            DebugInfoList.Clear();

            CvInvoke.Imshow(window, image);
        }

        public static float Norm(float min, float max, float val) => (val - min) / (max - min);
        public static double Norm(double min, double max, double val) => (val - min) / (max - min);

        public static float Lerp(float v0, float v1, float t) => (1.0f - t) * v0 + t * v1;
        public static double Lerp(double v0, double v1, double t) => (1.0 - t) * v0 + t * v1;

        public static double DistPointToLine(Point p1, Point p2, PointF p) => DistPointToLine(new LineSegment2D(p1, p2), p);

        public static double DistPointToLine(LineSegment2D l1, PointF p) => (double)(Math.Abs((l1.P2.Y - l1.P1.Y) * p.X - (l1.P2.X - l1.P1.X) * p.Y + l1.P2.X * l1.P1.Y - l1.P2.Y * l1.P1.X)
                                                                            / Math.Sqrt(Math.Pow(l1.P2.Y - l1.P1.Y, 2) + Math.Pow(l1.P2.X - l1.P1.X, 2)));

        public static PointF LineCenter(LineSegment2D l)
        {
            float dx = (float)(l.Direction.X < 0.0f ? l.P2.X - l.P1.X : l.P1.X - l.P2.X) * 0.5f;
            float dy = (float)(l.Direction.Y < 0.0f ? l.P2.Y - l.P1.Y : l.P1.Y - l.P2.Y) * 0.5f;

            return new PointF((float)(l.P1.X + dx), (float)(l.P2.Y + dy));
        }

        public static Point ApproxBallToPixelPos(RotatedRect screenBounds, PointF point, int resWi, int resHe)
        {
            var rect = screenBounds.MinAreaRect();
            double screenCorrectionOffset = 80.0; // 80 to correct for the top bar not being detected

            double distHor = point.X - rect.X;
            double normHor = Helpers.Norm(0, rect.Width, distHor);

            double distVer = point.Y - rect.Y;
            double normVer = Helpers.Norm(0, rect.Height, distVer);

            return new Point((int)(resWi * normHor), (int)(resHe * normVer + screenCorrectionOffset));
        }
    }

    class BasketAI
    {
        private Capture _capture;
        private System.Timers.Timer _timer;
        private string _title;
        private Process _adbProc;
        private ProcessStartInfo _adbInfo;
        private Tesseract _ocr;

        private int _points = 0;
        private float _interval = 1000 / 60;
        private float _fps = 60;

        private RotatedRect _currScreenRect;
        private CircleF _currCircle;
        private LineSegment2DF _currBasket;
        private Point _currBasketCenter;

        private bool _foundCircle = false;
        private bool _foundScreen = false;
        private bool _foundBasket = false;

        // phone resolution
        private const int ScreenResWidth = 1330;
        private const int ScreenResHeight = 2560;

        public BasketAI()
        {
            _capture = new Capture();
            _title = "Basked Ball";

            // DoSwipe(750, 2330, 730, 1080, 100);
        }

        public void Start()
        {
            CvInvoke.NamedWindow(_title, NamedWindowType.Normal);
            CvInvoke.NamedWindow("awd", NamedWindowType.Normal);

            var tessdataPath = @"D:\Emgu\emgucv-windesktop 3.1.0.2282\Emgu.CV.World\tessdata";
            _ocr = new Tesseract(tessdataPath, "eng", OcrEngineMode.TesseractCubeCombined);
            _ocr.SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ-1234567890"); //ABCDEFGHIJKLMNOPQRSTUVWXYZ-

            lastSignal = DateTime.Now;
            predictor = new PredictorModule();

            _timer = new System.Timers.Timer();
            _timer.Interval = _interval;
            _timer.Elapsed += DetectOnce;

            _timer.AutoReset = false;

            watch = new Stopwatch();
            watch.Reset();
            watch.Restart();
            lastTime = new TimeSpan(0, 0, 0, 0, 0);

            frameTimer = new Stopwatch();
            frameTimer.Restart();

            _timer.Start();

            CvInvoke.WaitKey(0);
            CvInvoke.DestroyWindow(_title);
        }


        public Point ApproxBallPixelPos(RotatedRect rect, PointF ballCenter, int resWi, int resHe)
        {
            var distHor = ballCenter.X - rect.MinAreaRect().X;
            var normHor = Helpers.Norm(0, rect.MinAreaRect().Width, distHor);

            var distVer = ballCenter.Y - rect.MinAreaRect().Y;
            var normVer = Helpers.Norm(0, rect.MinAreaRect().Height, distVer);

            return new Point((int)(resWi * normHor), (int)(resHe * normVer + 80)); // 80 to correct for the top bar not being detected
        }

        public void DoSwipe(int x0, int y0, int x1, int y1, int msDurration)
        {
            if (_adbProc == null || _adbInfo == null)
            {
                _adbProc = new Process();
                _adbInfo = new ProcessStartInfo();
                _adbInfo.WindowStyle = ProcessWindowStyle.Hidden;
                _adbInfo.FileName = "adb.exe";
            }

            _adbInfo.Arguments = String.Format("shell input swipe {0} {1} {2} {3} {4}", x0, y0, x1, y1, msDurration);
            _adbProc.StartInfo = _adbInfo;

            _adbProc.Start();
        }

        public int GetPoints(UMat frame)
        {
            //CvInvoke.Threshold(imageFrame, thresh, 200, 255, ThresholdType.BinaryInv);
            // CvInvoke.CvtColor(frame, thresh, ColorConversion.Bgr2Gray);
            UMat resized = new UMat(frame,
                new Rectangle(
                    new Point(
                        _currScreenRect.MinAreaRect().X + 80,
                        _currScreenRect.MinAreaRect().Y + 220
                        ),
                    new Size(
                        110, //_currScreenRect.MinAreaRect().Width, 
                        110 //_currScreenRect.MinAreaRect().Height
                        )));
            UMat thresh = new UMat();
            CvInvoke.Threshold(resized, thresh, 200, 255, ThresholdType.Binary);

            try
            {

                var builder = new StringBuilder();
                // CvInvoke.Imshow("awd", thresh);

                _ocr.Recognize(thresh);
                Tesseract.Character[] words = _ocr.GetCharacters();

                if (words.Length == 0) return -1;

                for (int i = 0; i < words.Length; i++)
                {
                    builder.Append(words[i].Text);
                }

                string pointStr = builder.ToString();
                pointStr = pointStr.Replace("O", "0");
                //Console.WriteLine("shit: " + pointStr);
                return Convert.ToInt32(pointStr);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }

            return -1;
        }

        public PointF GetBallVelocity()
        {
            var ret = new PointF(0f, 0f);

            if (arrPrevBasketPos.Count == 0)
                return ret;

            PointF avg = new PointF();
            for (int i = 0; i < arrPrevBasketPos.Count; i++)
            {
                avg.X += (arrPrevBasketPos[i].X);
                avg.Y += (arrPrevBasketPos[i].Y);
            }

            avg.X /= arrPrevBasketPos.Count;
            avg.Y /= arrPrevBasketPos.Count;

            avg.X /= _interval;
            avg.Y /= _interval;

            // Console.WriteLine(avg.X);

            if (arrPrevBasketPos.Count > 2)
                arrPrevBasketPos.RemoveAt(0);

            return avg;
        }

        public void DebugInfo(List<string> list, string fmt, params object[] args) => list.Add(string.Format(fmt, args));

        long ms = 0;
        int calibrationCount = 10;
        int calibrationTracker = 0;
        int maxDist = 1000;
        double flightTime = 0.7; // seconds
        PointF velBasket = new PointF(0f, 0f);
        Point currBasketCenter;
        Point prevBasketCenter;
        List<PointF> arrPrevBasketPos = new List<PointF>();

        int maxVertPos = 0;
        int minVertPos = 0;
        PointF currVelocity = new PointF(0f, 0f);
        PointF realPos;
        DateTime lastSignal;

        PointF tlLast;
        Stopwatch watch;
        TimeSpan lastTime;
        Stopwatch frameTimer;
        double msTime = 0.0;
        PredictorModule predictor;

        public void DetectOnce(object sender, ElapsedEventArgs e)
        {
            _timer.Stop();
            frameTimer.Stop();
            Console.WriteLine((frameTimer.Elapsed.TotalMilliseconds - lastTime.TotalMilliseconds));
            msTime = (frameTimer.Elapsed.TotalMilliseconds - lastTime.TotalMilliseconds);
            lastTime = frameTimer.Elapsed;
            frameTimer.Restart();
            watch.Restart();

            DoTick(msTime);
              
            watch.Stop();
            //Console.WriteLine(watch.ElapsedMilliseconds);
            lastTime = watch.Elapsed;
            _timer.Start();
        }

        List<string> debug = new List<string>();

        private void DoTick(double deltaTime)
        {

            Helpers.DebugInfo("delta ms: {0}", deltaTime);

            Mat imageFrame = _capture.QueryFrame();

            //TimeSpan delta = e.SignalTime - lastSignal;
            //lastSignal = e.SignalTime;
            //Console.WriteLine(delta.Milliseconds);
            //pictureBox.Image = imageFrame.Bitmap;

            /*
            UMat hsvImg = new UMat();
            CvInvoke.CvtColor(imageFrame, hsvImg, ColorConversion.Bgr2Hsv);

            Mat lowerRedHueRange = hsvImg.
            Mat upperRedHueRange = new Mat();

            CvInvoke.InRange(hsvImg, new Hsv(0,0,0), new Hsv(0,0,0), lowerRedHueRange);
            */
            

            //
            // Experimenting with more accurate detection of the orange line that is the basket
            //

            var min = new Hsv((double)(0.40 * 255), (double)(0.50 * 255), (double)(0.30 * 255));
            var max = new Hsv((double)(1.00 * 255), (double)(1.00 * 255), (double)(1.00 * 255));
            Image<Gray, Byte> tresh = imageFrame.ToImage<Hsv, Byte>().InRange(min, max);

            UMat treshGaussian = new UMat();
            CvInvoke.GaussianBlur(tresh, treshGaussian, new Size(3,3), 1,1);

            UMat threshCanny = new UMat();
            CvInvoke.Canny(treshGaussian, threshCanny, 120, 160);

            LineSegment2D[] tlines = CvInvoke.HoughLinesP(
                threshCanny,
                1, //Distance resolution in pixel-related units
                Math.PI / 45.0, //Angle resolution measured in radians.
                20, //threshold
                35, //min Line width
                15); //gap between lines


            float tlAvgX = 0;
            float tlAvgY = 0;
            int tlCount = 0;

            Image<Bgr, Byte> threshBgr = threshCanny.ToImage<Bgr, Byte>();
            foreach (LineSegment2D tl in tlines)
                if ((int)Math.Abs(Math.Round((decimal)tl.Direction.X)) == 1)
                {
                    threshBgr.Draw(tl, new Bgr(Color.DarkRed), 3);

                    PointF center = Helpers.LineCenter(tl);
                    var rect = _currScreenRect.MinAreaRect();

                    // horizontal bounds
                    LineSegment2D seg = new LineSegment2D(new Point(rect.X, rect.Y), new Point(rect.X + rect.Width, rect.Y));
                    var dist = Helpers.DistPointToLine(seg, center);
                    var ratio = Helpers.Norm(rect.Y, rect.Y + rect.Height, dist);

                    // vertical bounds
                    LineSegment2D vertSeg = new LineSegment2D(new Point(rect.X, rect.Y + rect.Height), new Point(rect.X, rect.Y));
                    var vertDist = Helpers.DistPointToLine(seg, center);
                    var vertRatio = Math.Abs(Helpers.Norm(rect.X, rect.X + rect.Width, vertDist));

                    if (ratio > 0.2 && ratio < 0.5 && vertRatio > 0.0 && vertRatio < 1.0)
                    {
                        //Console.WriteLine(vertDist);
                        //Console.WriteLine(vertRatio);

                        //CvInvoke.Circle(threshBgr, new Point((int)center.X, (int)center.Y), 5, new Bgr(0, 0, 255).MCvScalar, 3, LineType.FourConnected);
                        threshBgr.Draw(tl, new Bgr(Color.Cyan), 2);


                        {
                            tlAvgX += center.X;
                            tlAvgY += center.Y;
                            //avgCenterX /= 2;
                            //avgCenterY /= 2;
                            tlCount++;
                        }

                        //Console.WriteLine(line.Length);
                    }
                }

            tlAvgX /= tlCount;
            tlAvgY /= tlCount;
            tlLast = new PointF(tlAvgX, tlAvgY);

            //if (float.IsNaN(tlLast.X) || float.IsNaN(tlLast.Y))
            //    tlLast = new PointF(1, 1);

            /*
            float l0 = Helpers.Lerp(tlLast.Y, tlAvgX, 0.9f);
            float l1 = Helpers.Lerp(tlLast.Y, tlAvgY, 0.9f);
            l0 = float.IsNaN(l0) ? 1 : l0;
            l1 = float.IsNaN(l1) ? 1 : l1;

            tlLast = new PointF(l0, l1);

            */
            CvInvoke.Circle(threshBgr, tlLast.ToPoint(), 5, new Bgr(255, 0, 0).MCvScalar, 3, LineType.FourConnected);

            CvInvoke.Imshow("awd", threshBgr);
            

            // Grayscale
            UMat grayImg = new UMat();
            CvInvoke.CvtColor(imageFrame, grayImg, ColorConversion.Bgr2Gray);

            UMat pyr = new UMat();
            CvInvoke.PyrDown(grayImg, pyr);
            CvInvoke.PyrUp(pyr, grayImg);



            ms += (int)deltaTime;
            if (ms > 1300)
            {
                ms = 0;
                var pixPos = ApproxBallPixelPos(_currScreenRect, _currCircle.Center, ScreenResWidth, ScreenResHeight);
                var basketPixPos = ApproxBallPixelPos(_currScreenRect, new PointF((float)_currBasketCenter.X, (float)_currBasketCenter.Y), ScreenResWidth, ScreenResHeight);

                var bestPoint2 = ApproxBallPixelPos(_currScreenRect, realPos, ScreenResWidth, ScreenResHeight);

#if true
                if (_currBasketCenter.IsEmpty)
                    DoSwipe(pixPos.X, pixPos.Y, 730, 750, 100);
                else if (_points >= 10)
                {
                    DoSwipe(pixPos.X, pixPos.Y, bestPoint2.X, bestPoint2.Y, 100);
                }
                else
                    DoSwipe(pixPos.X, pixPos.Y, basketPixPos.X, basketPixPos.Y, 100);
#endif           

                UMat matmat = new UMat();
                grayImg.CopyTo(matmat);

                _points = GetPoints(matmat.Clone());

                //UMat matmat = new UMat();
                //grayImg.CopyTo(matmat);
            }
            
            
            
            //pictureBox2.Image = grayImg.Bitmap;

            // Circles
            double cannyThreshold = 180.0;
            double circleAccumThreshold = 130.0;
            CircleF[] circles = CvInvoke.HoughCircles(grayImg, HoughType.Gradient, 2.0, 5.0, cannyThreshold, circleAccumThreshold, 5);

            // Lines
            double cannyThresholdLinking = 130.0;
            UMat cannyEdges = new UMat();
            CvInvoke.Canny(grayImg, cannyEdges, 120, cannyThresholdLinking);
            // CvInvoke.Imshow("awd", grayImg);

            LineSegment2D[] lines = CvInvoke.HoughLinesP(
              cannyEdges,
              1, //Distance resolution in pixel-related units
              Math.PI / 45.0, //Angle resolution measured in radians.
              25, //threshold
              40, //min Line width
              20); //gap between lines


            // Rects and triangles
            VectorOfVectorOfPoint contoursDetected = new VectorOfVectorOfPoint();
            CvInvoke.FindContours(grayImg, contoursDetected, null, RetrType.List, ChainApproxMethod.ChainApproxSimple);

            var triangles = new List<Triangle2DF>();
            var rectangles = new List<RotatedRect>();

            using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
            {
                CvInvoke.FindContours(cannyEdges, contours, null, RetrType.List, ChainApproxMethod.ChainApproxSimple);
                int count = contours.Size;
                for (int i = 0; i < count; i++)
                {
                    using (VectorOfPoint contour = contours[i])
                    using (VectorOfPoint approxContour = new VectorOfPoint())
                    {
                        CvInvoke.ApproxPolyDP(contour, approxContour, CvInvoke.ArcLength(contour, true) * 0.02, true);
                        if (CvInvoke.ContourArea(approxContour, false) > 10000) //only consider contours with area greater than 250
                        {
                            if (approxContour.Size == 3) //The contour has 3 vertices, it is a triangle
                            {
                                Point[] pts = approxContour.ToArray();
                                triangles.Add(new Triangle2DF(
                                   pts[0],
                                   pts[1],
                                   pts[2]
                                   ));
                            }
                            else if (approxContour.Size == 4) //The contour has 4 vertices.
                            {
#region determine if all the angles in the contour are within [80, 100] degree
                                bool isRectangle = true;
                                Point[] pts = approxContour.ToArray();
                                LineSegment2D[] edges = PointCollection.PolyLine(pts, true);

                                for (int j = 0; j < edges.Length; j++)
                                {
                                    double angle = Math.Abs(
                                       edges[(j + 1) % edges.Length].GetExteriorAngleDegree(edges[j]));
                                    if (angle < 85 || angle > 95)
                                    {
                                        isRectangle = false;
                                        break;
                                    }
                                }
#endregion

                                if (isRectangle) rectangles.Add(CvInvoke.MinAreaRect(approxContour));
                            }
                        }
                    }
                }
            }


            // draw all the things 

            Image<Bgr, Byte> circleImg = imageFrame.ToImage<Bgr, Byte>();

            //foreach (CircleF c in circles)
            //{
            if (circles.Length > 0)
            {
                // add some guarding if some other circle pops up
                _currCircle = circles[0];

                if (!_foundCircle)
                    _foundCircle = true;
            }

            if (_foundCircle)
            {
                circleImg.Draw(_currCircle, new Bgr(Color.Blue), 2);
                CvInvoke.Circle(circleImg, new Point((int)_currCircle.Center.X, (int)_currCircle.Center.Y), 5, new Bgr(255, 0, 0).MCvScalar, 3, LineType.FourConnected);

                var pixPos = ApproxBallPixelPos(_currScreenRect, _currCircle.Center, ScreenResWidth, ScreenResHeight);
                Helpers.DebugInfo("Circle pix-pos: ({0}, {1})", pixPos.X, pixPos.Y);
            }

            foreach (Triangle2DF triangle in triangles)
                circleImg.Draw(triangle, new Bgr(Color.DarkBlue), 2);

            // foreach (RotatedRect box in rectangles)
            //    circleImg.Draw(box, new Bgr(Color.DarkOrange), 2);

            if (rectangles.Count > 0 )
            {
                RotatedRect biggest = rectangles[0];
                foreach (RotatedRect rect in rectangles)
                    if (rect.Size.Width * rect.Size.Height > biggest.Size.Width * biggest.Size.Height)
                        biggest = rect;

                // the biggest is probably the screen
                _currScreenRect = biggest;

                //Console.WriteLine("Biggest rectangle size: {0}, w: {1}, h: {2}", biggest.Size.Width * biggest.Size.Height, biggest.Size.Width, biggest.Size.Height);
                if (!_foundScreen)
                    _foundScreen = true;
            }

            if (_foundScreen)
            {
                string rectangleText = String.Format("Screen rect: ({0}, {1}), ({2}, {3})",
                    _currScreenRect.MinAreaRect().X.ToString(),
                    _currScreenRect.MinAreaRect().Y.ToString(),
                    ((int)(_currScreenRect.MinAreaRect().X + _currScreenRect.Size.Width)).ToString(),
                    ((int)(_currScreenRect.MinAreaRect().Y + _currScreenRect.Size.Height)).ToString());
               Helpers.DebugInfo(rectangleText);

                circleImg.Draw(_currScreenRect, new Bgr(Color.Aqua), 3);
            }

            float avgCenterX = 0;
            float avgCenterY = 0;
            int count2 = 0;
            var lineSuggestions = new List<LineSegment2D>();
            foreach (LineSegment2D line in tlines)
            {

                //Console.WriteLine((int)Math.Abs(Math.Round((decimal)line.Direction.X)));
                if ((int)Math.Abs(Math.Round((decimal)line.Direction.X)) == 1)
                {
                    PointF center = Helpers.LineCenter(line);
                    var rect = _currScreenRect.MinAreaRect();
                    
                    // horizontal bounds
                    LineSegment2D seg = new LineSegment2D(new Point(rect.X, rect.Y), new Point(rect.X + rect.Width, rect.Y));
                    var dist = Helpers.DistPointToLine(seg, center);
                    var ratio = Helpers.Norm(rect.Y, rect.Y + rect.Height, dist);

                    // vertical bounds
                    LineSegment2D vertSeg = new LineSegment2D(new Point(rect.X, rect.Y + rect.Height), new Point(rect.X, rect.Y));
                    var vertDist = Helpers.DistPointToLine(seg, center);
                    var vertRatio = Math.Abs(Helpers.Norm(rect.X, rect.X + rect.Width, vertDist));

                    if (ratio > 0.2 && ratio < 0.5 && vertRatio > 0.0 && vertRatio < 1.0)
                    {
                        //Console.WriteLine(vertDist);
                        //Console.WriteLine(vertRatio);

                        lineSuggestions.Add(line);

                        CvInvoke.Circle(circleImg, new Point((int)center.X, (int)center.Y), 5, new Bgr(0, 0, 255).MCvScalar, 3, LineType.FourConnected);
                        circleImg.Draw(line, new Bgr(Color.Cyan), 2);

                        
                        {
                            avgCenterX += center.X;
                            avgCenterY += center.Y;
                            //avgCenterX /= 2;
                            //avgCenterY /= 2;
                            count2++;
                        }

                        //Console.WriteLine(line.Length);
                    }
                }
            }

            avgCenterX /= count2;
            avgCenterY /= count2;

            /*
            if (lineSuggestions.Count > 0)
            {
                PointF highestLine = Helpers.LineCenter(lineSuggestions[0]);
                foreach (var line in lineSuggestions)
                {
                    PointF center = Helpers.LineCenter(line);
                    if (center.Y < highestLine.Y)
                        highestLine = center;
                }

                avgCenterX += highestLine.X;
                avgCenterY += highestLine.Y;
                avgCenterX /= 2;
                avgCenterY /= 2;

                CvInvoke.Circle(circleImg, new Point((int)highestLine.X, (int)highestLine.Y), 5, new Bgr(Color.Cornsilk).MCvScalar, 3, LineType.FourConnected);
                //circleImg.Draw(, new Bgr(Color.Green), 2);
            }
            else if (lineSuggestions.Count != 0)
            {
                PointF highestLine = Helpers.LineCenter(lineSuggestions[0]);
                avgCenterX += highestLine.X;
                avgCenterY += highestLine.Y;
                avgCenterX /= 2;
                avgCenterY /= 2;
            }*/


            //if (calibrationTracker < calibrationCount)
            //calibrationTracker++;
            //else if (calibrationTracker >= calibrationCount)
            //maxDist = 100;

            // if (Math.Abs(_currBasketCenter.X - avgCenterX) < maxDist && Math.Abs(_currBasketCenter.X - avgCenterY) < maxDist)

            //_currBasketCenter = new Point((int)Norm(avgCenterX, _currBasketCenter.X, 0.5f), (int)Norm(avgCenterY, _currBasketCenter.Y, 0.5f));

            //if (_currBasketCenter.X == 0 && _currBasketCenter.Y == 0)
            bool avgInScreen = _currScreenRect.MinAreaRect().IntersectsWith(new Rectangle((int)avgCenterX, (int)avgCenterY, 1, 1));
            int predictedDirection = 0;

            if (_foundScreen && _foundCircle)
            {
                if (avgInScreen)
                    _currBasketCenter = new Point((int)Helpers.Lerp((float)_currBasketCenter.X, (float)avgCenterX, 0.45f), (int)Helpers.Lerp((float)_currBasketCenter.Y, (float)avgCenterY, 0.45f));
                //else
                //    _currBasketCenter = new Point((int)Norm(_currBasketCenter.X, avgCenterX, 0.5f), (int)Norm(_currBasketCenter.Y, avgCenterY, 0.5f));
                CvInvoke.Circle(circleImg, _currBasketCenter, 10, new Bgr(255, 0, 255).MCvScalar, 2, LineType.FourConnected);

                Helpers.DebugInfo("basketPos: ({0}, {1})", _currBasketCenter.X, currBasketCenter.Y);
                predictor.AddTickEntry(_currBasketCenter, deltaTime, _currScreenRect);
                PointF prediction = predictor.GetNextPrediction();
                realPos = prediction;

                CvInvoke.Line(circleImg, _currCircle.Center.ToPoint(), prediction.ToPoint(), Color.Orange.ToScalar(), 2);

                predictedDirection = predictor.GetDirection();
                Helpers.DebugInfo("Direction: {0}", predictedDirection);
            }

            velBasket = GetBallVelocity();
            velBasket.X -= 21.0f;
            velBasket.Y -= 11.25f;

            if (Math.Abs(velBasket.X) > currVelocity.X)
                currVelocity.X = Math.Abs(velBasket.X);
            if (Math.Abs(velBasket.Y) > currVelocity.Y)
                currVelocity.Y = Math.Abs(velBasket.Y);

            if (avgInScreen)
                arrPrevBasketPos.Add(_currBasketCenter);

            var rect2 = _currScreenRect.MinAreaRect();
            LineSegment2D vertSeg2 = new LineSegment2D(new Point(rect2.X + rect2.Width, rect2.Y), new Point(rect2.X + rect2.Width, rect2.Y + rect2.Height));
            var vertDist2 = Helpers.DistPointToLine(vertSeg2, _currBasketCenter);
            var vertRatio2 = Math.Abs(Helpers.Norm(rect2.X, rect2.X + rect2.Width, vertDist2));

            if (minVertPos == 0) minVertPos = 164; //(int)vertDist2;
            if (maxVertPos == 0) maxVertPos = 164; //(int)vertDist2;

            if (vertDist2 > maxVertPos && vertDist2 < 230)
                maxVertPos = (int)vertDist2;

            if (vertDist2 < minVertPos)
                minVertPos = (int)vertDist2;

            //realPos = new PointF(0, 0); 
            //if (velBasket.X > 0)
            //realPos = new PointF((float)_currBasketCenter.X + (float)(velBasket.X * 0.7 * _interval), (float)_currBasketCenter.Y);
            //else
                //realPos = new PointF((float)_currBasketCenter.X + (float)(velBasket.X * 0.7 * _interval), (float)_currBasketCenter.Y);
            //var bestPoint = ApproxBallPixelPos(_currScreenRect, realPos, ScreenResWidth, ScreenResHeight);

            //var pixPos = ApproxBallPixelPos(_currScreenRect, _currCircle.Center, ScreenResWidth, ScreenResHeight);
            //var basketPixPos = ApproxBallPixelPos(_currScreenRect, new PointF((float)_currBasketCenter.X, (float)_currBasketCenter.Y), ScreenResWidth, ScreenResHeight);

            //CvInvoke.Line(circleImg, new Point((int)_currCircle.Center.X, (int)_currCircle.Center.X), bestPoint, new Bgr(0, 255, 0).MCvScalar, 2);
            CvInvoke.Line(circleImg, _currCircle.Center.ToPoint(), _currBasketCenter, new Bgr(0, 255, 0).MCvScalar, 2);
            //CvInvoke.Line(circleImg, _currCircle.Center.ToPoint(), realPos.ToPoint(), Color.Red.ToScalar(), 2);

            //PointF basketPosDelta = new PointF((float)(_currBasketCenter.X - prevBasketCenter.X), (float)(_currBasketCenter.Y - prevBasketCenter.Y));
            //velBasket = basketPosDelta;
            //prevBasketCenter = _currBasketCenter;

            //CvInvoke.Circle(circleImg, _currBasketCenter)


            if (_foundScreen && _foundCircle)
            {
                Helpers.DebugInfo("Points: {0}", _points);
            }

            /*CvInvoke.PutText(circleImg, String.Format("BasketVel: ({0}, {1})", velBasket.X, velBasket.Y), new Point(10, 65), FontFace.HersheySimplex, 0.5, new Bgr(0, 255, 0).MCvScalar, 1);
            CvInvoke.PutText(circleImg, String.Format("VertRatio: ({0}, {1})", vertDist2, vertRatio2), new Point(10, 80), FontFace.HersheySimplex, 0.5, new Bgr(0, 255, 0).MCvScalar, 1);
            CvInvoke.PutText(circleImg, String.Format("min/max: ({0}, {1})", minVertPos, maxVertPos), new Point(10, 95), FontFace.HersheySimplex, 0.5, new Bgr(0, 255, 0).MCvScalar, 1);
            CvInvoke.PutText(circleImg, String.Format("const vel: ({0}, {1})", currVelocity.X, currVelocity.Y), new Point(10, 110), FontFace.HersheySimplex, 0.5, new Bgr(0, 255, 0).MCvScalar, 1);
            */

            Helpers.DebugInfo("BasketVel: ({0}, {1})", velBasket.X, velBasket.Y);
            Helpers.DebugInfo("VertRatio: ({0}, {1})", vertDist2, vertRatio2);
            Helpers.DebugInfo("min/max: ({0}, {1})", minVertPos, maxVertPos);
            Helpers.DebugInfo("const vel: ({0}, {1})", currVelocity.X, currVelocity.Y);

            // CvInvoke.PutText(circleImg, "Test", new Point(10, 100), FontFace.HersheySimplex, 1, new Bgr(0, 255, 0).MCvScalar, 1);

            //CvInvoke.Imshow(_title, circleImg);

            predictor.AddDebugInfo(circleImg);

            Helpers.AddDebugInfoAndDraw(_title, circleImg);

            /*
            int startX = 5;
            int startY = 15;
            int deltaY = 20;

            for (int i = 0; i < debug.Count; i++)
            {
                CvInvoke.PutText(circleImg, debug[i], new Point(startX, (int)(startY + (deltaY * i))), FontFace.HersheySimplex, 0.5, new Bgr(0, 255, 0).MCvScalar, 1);
            }

            //DebugInfoList.Clear();
            debug = new List<string>();

            CvInvoke.Imshow(_title, circleImg);
            */

            /*
            var min = new Hsv(8, 51, 99);
            var max = new Hsv(8, 100, 69);
            Image<Gray, Byte> tresh = imageFrame.ToImage<Hsv, Byte>().InRange(min, max);
            CvInvoke.Imshow("thresh", tresh);
            */

            //pictureBox2.Image = circleImg.ToBitmap();
        }
    }
}
