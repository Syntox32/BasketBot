using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;

using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.Util;
using Emgu.CV.CvEnum;
using Emgu.CV.Util;
using System.Threading;
using System.Timers;
using System.Diagnostics;
using System.Windows.Input;

namespace Basket
{
    class Program
    {
        static void Main(string[] args)
        {
            var ai = new BasketAI();
            ai.Start();
        }
    }

    class BasketAI
    {
        private Capture _capture;
        private System.Timers.Timer _timer;
        private string _title;
        private Process _adbProc;
        private ProcessStartInfo _adbInfo;

        private RotatedRect _currScreenRect;
        private CircleF _currCircle;
        private LineSegment2DF _currBasket;

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
            // CvInvoke.NamedWindow("thresh", NamedWindowType.Normal);

            _timer = new System.Timers.Timer();
            _timer.Interval = 1000 / 60;
            _timer.Elapsed += DetectOnce;

            _timer.Start();

            CvInvoke.WaitKey(0);
            CvInvoke.DestroyWindow(_title);
        }

        public float DistPointToLine(Point p1, Point p2, PointF p)
        {
            return DistPointToLine(new LineSegment2D(p1, p2), p);
        }

        public float DistPointToLine(LineSegment2D l1, PointF p)
        {
            return (float)(Math.Abs((l1.P2.Y - l1.P1.Y) * p.X - (l1.P2.X - l1.P1.X) * p.Y + l1.P2.X * l1.P1.Y - l1.P2.Y * l1.P1.X)
                / Math.Sqrt(Math.Pow(l1.P2.Y - l1.P1.Y, 2) + Math.Pow(l1.P2.X - l1.P1.X, 2)));
        }

        public PointF LineCenter(LineSegment2D l)
        {
            var dx = (float)(l.Direction.X < 0.0f ? l.P2.X - l.P1.X : l.P1.X - l.P2.X) * 0.5;
            var dy = (float)(l.Direction.Y < 0.0f ? l.P2.Y - l.P1.Y : l.P1.Y - l.P2.Y) * 0.5;

            return new PointF((float)(l.P1.X + dx), (float)(l.P2.Y + dy));
        }

        public float Norm(float min, float max, float val)
        {
            return (val - min) / (max - min);
        }

        public Point ApproxBallPixelPos(RotatedRect rect, PointF ballCenter, int resWi, int resHe)
        {
            var distHor = ballCenter.X - rect.MinAreaRect().X;
            var normHor = Norm(0, rect.MinAreaRect().Width, distHor);

            var distVer = ballCenter.Y - rect.MinAreaRect().Y;
            var normVer = Norm(0, rect.MinAreaRect().Height, distVer);

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

        long ms = 0;

        public void DetectOnce(object sender, EventArgs e)
        {
            ms += (int)(1000 / 60);
            if (ms > 1500)
            {
                ms = 0;
                var pixPos = ApproxBallPixelPos(_currScreenRect, _currCircle.Center, ScreenResWidth, ScreenResHeight);
                DoSwipe(pixPos.X, pixPos.Y, 730, 750, 100);
            }

            Mat imageFrame = _capture.QueryFrame();
            //pictureBox.Image = imageFrame.Bitmap;


            // Grayscale
            UMat grayImg = new UMat();
            CvInvoke.CvtColor(imageFrame, grayImg, ColorConversion.Bgr2Gray);

            UMat pyr = new UMat();
            CvInvoke.PyrDown(grayImg, pyr);
            CvInvoke.PyrUp(pyr, grayImg);

            //pictureBox2.Image = grayImg.Bitmap;

            // Circles
            double cannyThreshold = 180.0;
            double circleAccumThreshold = 130.0;
            CircleF[] circles = CvInvoke.HoughCircles(grayImg, HoughType.Gradient, 2.0, 5.0, cannyThreshold, circleAccumThreshold, 5);

            // Lines
            double cannyThresholdLinking = 130.0;
            UMat cannyEdges = new UMat();
            CvInvoke.Canny(grayImg, cannyEdges, 120, cannyThresholdLinking);
            //CvInvoke.Imshow(_title, cannyEdges);

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
                CvInvoke.PutText(circleImg, String.Format("Circle pix-pos: ({0}, {1})", pixPos.X, pixPos.Y), new Point(10, 30), FontFace.HersheySimplex, 0.5, new Bgr(0, 255, 0).MCvScalar, 1);
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
                CvInvoke.PutText(circleImg, rectangleText, new Point(10, 15), FontFace.HersheySimplex, 0.5, new Bgr(0, 255, 0).MCvScalar, 1);

                circleImg.Draw(_currScreenRect, new Bgr(Color.Aqua), 3);
            }

            foreach (LineSegment2D line in lines)
            {

                //Console.WriteLine((int)Math.Abs(Math.Round((decimal)line.Direction.X)));
                if ((int)Math.Abs(Math.Round((decimal)line.Direction.X)) == 1)
                {
                    PointF center = LineCenter(line);
                    var rect = _currScreenRect.MinAreaRect();
                    LineSegment2D seg = new LineSegment2D(new Point(rect.X, rect.Y), new Point(rect.X + rect.Width, rect.Y));
                    var dist = DistPointToLine(seg, center);
                    var ratio = Norm(rect.Y, rect.Y + rect.Height, dist);

                    if (ratio > 0.2 && ratio < 0.5)
                    {
                        Console.WriteLine(dist);
                        Console.WriteLine(ratio);



                        CvInvoke.Circle(circleImg, new Point((int)center.X, (int)center.Y), 5, new Bgr(0, 0, 255).MCvScalar, 3, LineType.FourConnected);
                        circleImg.Draw(line, new Bgr(Color.Green), 2);
                        //Console.WriteLine(line.Length);
                    }
                }
            }

            // CvInvoke.PutText(circleImg, "Test", new Point(10, 100), FontFace.HersheySimplex, 1, new Bgr(0, 255, 0).MCvScalar, 1);

            CvInvoke.Imshow(_title, circleImg);

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
