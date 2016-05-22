using System;
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
    public class CaptureModule
    {
        private Capture _capture;

        public CaptureModule()
        {
            try
            {
                _capture = new Capture();
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error initalizing capture instance: ", ex.Message);
            }
        }

        public Mat SingleCapture() => _capture.QueryFrame();
    }

    public class InteractionModule
    {
        private Process _adbProc;
        private ProcessStartInfo _adbInfo;

        public InteractionModule()
        {
            try
            {
                if (_adbProc == null || _adbInfo == null)
                {
                    _adbProc = new Process();
                    _adbInfo = new ProcessStartInfo();
                    _adbInfo.WindowStyle = ProcessWindowStyle.Hidden;
                    _adbInfo.FileName = "adb.exe";
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error initalizing InteractionModule: ", ex.Message);
            }
        }

        public void DoSwipe(int x0, int y0, int x1, int y1, int msDurration)
        {
            _adbInfo.Arguments = string.Format("shell input swipe {0} {1} {2} {3} {4}", x0, y0, x1, y1, msDurration);
            _adbProc.StartInfo = _adbInfo;

            _adbProc.Start();
        }
    }

    public struct PredictionEntry
    {
        public PredictionEntry(PointF p, double ms)
        {
            this.Point = p;
            this.MsDelta = ms;
        }

        public PointF Point { get; }
        public double MsDelta { get; }
    }

    public class PredictorModule
    {
        private const int MaxEntries = 100;
        private const double BallFlightTime = 0.7; // ms

        private List<PredictionEntry> _entries;
        private PointF _posMax;
        private PointF _posMin;

        public PointF BasketVelocity { get; private set; } = new PointF(0, 0);

        public PredictorModule()
        {
            _entries = new List<PredictionEntry>();
        }

        public void AddTickEntry(PointF basketCenter, double deltaTime, RotatedRect screenBounds)
        {
            bool inScreen = screenBounds.MinAreaRect().IntersectsWith(new Rectangle(basketCenter.ToPoint(), new Size(1, 1)));
            if (!inScreen)
                return;

            if (_entries.Count >= MaxEntries)
                _entries.RemoveAt(0); // if too many entries, remove the oldest one

            if (_posMax.IsEmpty) _posMax = new PointF(screenBounds.MinAreaRect().X + screenBounds.Size.Width, 0); //basketCenter;
            if (_posMin.IsEmpty) _posMin = new PointF(screenBounds.MinAreaRect().X + screenBounds.Size.Width, 0); // basketCenter;

            if (basketCenter.X > _posMax.X) _posMax = basketCenter;
            if (basketCenter.X < _posMin.X) _posMin = basketCenter;

            _entries.Add(new PredictionEntry(basketCenter, deltaTime));
        }

        private PredictionEntry PredictApproxVelocity()
        {
            float avgLength = 0.0f;
            double avgMs = 0.0;

            for (int i = 0; i < _entries.Count; i++)
            {
                float length = _entries[i].Point.Length();
                double ms = _entries[i].MsDelta;

                avgLength += length;
                avgMs += ms;
            }

            avgLength /= (float)avgMs; //_entries.Count;
            avgMs /= _entries.Count;

            return new PredictionEntry(new PointF(avgLength, 0f), avgMs);
        }

        public int GetDirection()
        {
            double direction = 0d;

            if (_entries.Count >= 10)
            {
                var first = _entries[_entries.Count - 1];
                var inter = _entries[_entries.Count - 8];

                var delta = first.Point.X - inter.Point.X;

                if (delta > 0)
                    return 1;
                else
                    return -1;
            }

            return 0;
        }

        public PointF GetNextPrediction()
        {
            if (_entries.Count == 0)
                return new Point(0, 0); //throw new InvalidOperationException("No entries to predict from.");

            PredictionEntry lastest = _entries[_entries.Count - 1];

            var vel = PredictApproxVelocity();
            var dir = GetDirection();

            return new PointF(lastest.Point.X + dir * (float)(5.0f * BallFlightTime * vel.Point.X), lastest.Point.Y);
        }

        public void AddDebugInfo(Image<Bgr, Byte> image)
        {
            Helpers.DebugInfo("max/min: ({0}, {1})", _posMax.X, _posMin.X);
            Helpers.DebugInfo("Entry count: {0}", _entries.Count);

            var pred = PredictApproxVelocity();
            Helpers.DebugInfo("Approx. vel: {0} /{1}", pred.Point.X, pred.MsDelta);
        }
    }

    public class DetectorModule
    {
        private Tesseract _ocr; // image recognition

        private UMat _grayImg;
        private UMat _cannyImg;

        private bool _invalidateContours;
        private bool _invalidateCircle;
        private bool _invalidateScreen;

        private bool _hasResolution;
        private int _resolutionWidth;
        private int _resolutionHeight;

        private CircleF _currCircle;
        private RotatedRect _currScreenBounds;

        // private List<CircleF> _circles;
        private List<Triangle2DF> _triangles;
        private List<RotatedRect> _rectangles;

        public bool FoundScreen { get; private set; }
        public bool FoundBall { get; private set; }
        public bool FoundBasket { get; private set; }

        public CircleF Ball { get { return _currCircle; } }
        public RotatedRect ScreenBounds { get { return _currScreenBounds; } }

        private const double MinContourArea = 10000.0;

        public DetectorModule()
        {
            FoundScreen = false;
            FoundBall = false;
            FoundBasket = false;

            _hasResolution = false;
            _invalidateContours = true;
            _invalidateCircle = true;
            _invalidateScreen = true;

            _triangles = new List<Triangle2DF>();
            _rectangles = new List<RotatedRect>();

            InitOCREngine(@"D:\Emgu\emgucv-windesktop 3.1.0.2282\Emgu.CV.World\tessdata");
        }

        public void InvalidateGrayFrame(IInputArray newFrame)
        {
            // Grayscale
            UMat grayImg = new UMat();
            CvInvoke.CvtColor(newFrame, grayImg, ColorConversion.Bgr2Gray);

            // Try to remove some noise
            UMat pyr = new UMat();
            CvInvoke.PyrDown(grayImg, pyr);
            CvInvoke.PyrUp(pyr, grayImg);

            _grayImg = grayImg;

            InvalidateCanny(_grayImg);

            // Tell the module to invalidate contours the next time it's requested
            _invalidateContours = true;
            _invalidateCircle = true;
            _invalidateScreen = true;
        }

        public RotatedRect GetScreen()
        {
            //if (_invalidateScreen || !FoundScreen)
            if (FindScreen(FindRectangles()))
                FoundScreen = true;

            return _currScreenBounds;
        }

        public CircleF GetBall()
        {
            GetBallCenter();

            return _currCircle;
        }

        public PointF GetBallCenter()
        {
            //if (_invalidateScreen || !FoundBall)
            if (FindBall(FindCircles(_grayImg)))
                FoundBall = true;

            return _currCircle.Center;
        }

        public PointF GetBasket()
        {
            return new PointF();
        }


        public int GetPointCount(UMat grayImg, RotatedRect screenBounds)
        {
            var intermediate = new UMat();
            grayImg.CopyTo(intermediate);

            // We can't do any operations on the same image instance passed in the params
            return DetectPointCount(intermediate.Clone(), screenBounds);
        }

        public void SetScreenResolution(int width, int height)
        {
            _resolutionWidth = width;
            _resolutionHeight = width;

            _hasResolution = true;
        }

        private void InitOCREngine(string tessdataPath)
        {
            try
            {
                _ocr = new Tesseract(tessdataPath, "eng", OcrEngineMode.TesseractCubeCombined);
                _ocr.SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ-1234567890");
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error initalizing OCR-engine: {0}", ex.Message);
            }
        }

        private void InvalidateCanny(UMat grayImg)
        {
            double cannyThreshold = 120.0;
            double cannyThresholdLinking = 130.0;

            UMat cannyEdges = new UMat();
            CvInvoke.Canny(grayImg, cannyEdges, cannyThreshold, cannyThresholdLinking);

            _cannyImg = cannyEdges;
        }

        private bool FindScreen(List<RotatedRect> rectangles)
        {
            var found = false;

            if (rectangles.Count > 0)
            {
                RotatedRect biggest = rectangles[0];

                //foreach (RotatedRect rect in rectangles)
                for (int i = 0; i < rectangles.Count; i++)
                {
                    if (rectangles[i].Size.Width * rectangles[i].Size.Height > biggest.Size.Width * biggest.Size.Height)
                        biggest = rectangles[i];
                }

                // the biggest is prooobably the screen
                // ...can't be sure though
                _currScreenBounds = biggest;

                if (!found)
                    found = true;

                // _invalidateScreen = false;

                return found;
            }

            return found;
        }

        private bool FindBall(CircleF[] circles)
        {
            var found = false;

            if (circles.Length > 0)
            {
                // TODO: add some guarding if some other circle pops up
                _currCircle = circles[0]; // just set the first circle as the ball

                if (!found)
                    found = true;

                _invalidateCircle = false;

                return found;
            }

            return found;
        }

        private CircleF[] FindCircles(UMat grayImg)
        {
            double cannyThreshold = 180.0;
            double circleAccumThreshold = 130.0;
            CircleF[] circles = CvInvoke.HoughCircles(grayImg, HoughType.Gradient, 2.0, 5.0, cannyThreshold, circleAccumThreshold, 5);

            return circles;
        }

        private LineSegment2D[] FindLines(UMat grayImg)
        {
            double cannyThreshold = 120.0;
            double cannyThresholdLinking = 130.0;

            UMat cannyEdges = new UMat();
            CvInvoke.Canny(grayImg, cannyEdges, cannyThreshold, cannyThresholdLinking);

            LineSegment2D[] lines = CvInvoke.HoughLinesP(
              cannyEdges,
              1, // Distance resolution in pixel-related units
              Math.PI / 45.0, // Angle resolution measured in radians.
              25,  // threshold
              40,  // min Line width
              20); // gap between lines

            return lines;
        }

        private List<RotatedRect> FindRectangles()
        {
            FindRectanglesAndTriangles(_grayImg);

            return _rectangles;
        }

        private List<Triangle2DF> FindTriangles()
        {
            FindRectanglesAndTriangles(_grayImg);

            return _triangles;
        }

        private void FindRectanglesAndTriangles(UMat grayImg)
        {
            //if (!_invalidateContours) return;

            _triangles = new List<Triangle2DF>();
            _rectangles = new List<RotatedRect>();

            // Rects and triangles
            var contoursDetected = new VectorOfVectorOfPoint();
            CvInvoke.FindContours(grayImg, contoursDetected, null, RetrType.List, ChainApproxMethod.ChainApproxSimple);

            using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
            {
                CvInvoke.FindContours(_cannyImg, contours, null, RetrType.List, ChainApproxMethod.ChainApproxSimple);
                int count = contours.Size;

                for (int i = 0; i < count; i++)
                {
                    using (VectorOfPoint contour = contours[i])
                    using (var approxContour = new VectorOfPoint())
                    {
                        // Approximate a polygonal curve
                        CvInvoke.ApproxPolyDP(contour, approxContour, CvInvoke.ArcLength(contour, true) * 0.02, true);

                        if (CvInvoke.ContourArea(approxContour, false) > MinContourArea) // Only consider contours with area greater than MinContourArea
                        {
                            if (approxContour.Size == 3) // The contour has 3 vertices, it is a triangle
                            {
                                Point[] pts = approxContour.ToArray();
                                _triangles.Add(new Triangle2DF(pts[0], pts[1], pts[2]));
                            }
                            else if (approxContour.Size == 4) // The contour has 4 vertices.
                            {
                                // Determine if all the angles in the contour are within [80, 100] degree
                                bool isRectangle = true;
                                Point[] pts = approxContour.ToArray();
                                LineSegment2D[] edges = PointCollection.PolyLine(pts, true);

                                for (int j = 0; j < edges.Length; j++)
                                {
                                    double angle = Math.Abs(edges[(j + 1) % edges.Length].GetExteriorAngleDegree(edges[j]));

                                    if (angle < 85 || angle > 95)
                                    {
                                        isRectangle = false;
                                        break;
                                    }
                                }

                                if (isRectangle) _rectangles.Add(CvInvoke.MinAreaRect(approxContour));
                            }
                        }
                    }
                }
            }

            _invalidateContours = false;
        }

        private int DetectPointCount(UMat grayImg, RotatedRect screenBounds)
        {
            // Crop out everything other than the point count
            UMat resized = new UMat(grayImg, new Rectangle(new Point(screenBounds.MinAreaRect().X + 80, screenBounds.MinAreaRect().Y + 220), new Size(110, 110)));

            UMat thresh = new UMat();
            CvInvoke.Threshold(resized, thresh, 200, 255, ThresholdType.Binary);

            var builder = new StringBuilder();

            try
            {
                _ocr.Recognize(thresh);
                Tesseract.Character[] words = _ocr.GetCharacters();

                if (words.Length == 0) return -1;

                for (int i = 0; i < words.Length; i++)
                {
                    builder.Append(words[i].Text);
                }

                string pointStr = builder.ToString();

                // Sometimes numbers are mistaken for look-a-like characters
                pointStr = pointStr.Replace("O", "0");

                int parsedCount = 0;
                bool succ = int.TryParse(pointStr, out parsedCount);

                return succ ? parsedCount : -1;
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }

            return -1;
        }
    }

    public class DisplayModule
    {
        public string Title { get; private set; }

        public DisplayModule(string title)
        {
            Title = title;

            CvInvoke.NamedWindow(Title, NamedWindowType.Normal);
            //CvInvoke.NamedWindow("awd", NamedWindowType.Normal);
        }

        public void Start()
        {
            CvInvoke.WaitKey(0);
            CvInvoke.DestroyWindow(Title);
        }

        public void Display(Image<Bgr, Byte> captureFrame)
        {
            Helpers.AddDebugInfoAndDraw(Title, captureFrame);
        }
    }

    public class AIModule
    {
        private DetectorModule detector;
        private InteractionModule interactor;
        private CaptureModule capture;
        private PredictorModule predictor;
        private DisplayModule display;

        private const int FPS = 60;
        private const double BallFlightTime = 0.7; // milliseconds
        private const double LoopInterval = 1000 / FPS;
        private const string windowTitle = "Basked Ball";

        // phone resolution
        private const int ScreenResWidth = 1330; // can't remember if this is an error or not, should be 1440, but it works so I'll leave it here
        private const int ScreenResHeight = 2560;

        private Timer loopTimer;
        private object loopLock;
        private double elapsed;

        private int currPointCount;
        private DateTime lastSignal;

        public AIModule()
        {
            capture = new CaptureModule();
            interactor = new InteractionModule();
            detector = new DetectorModule();
            predictor = new PredictorModule();
            display = new DisplayModule(windowTitle);

            detector.SetScreenResolution(ScreenResWidth, ScreenResHeight);

            elapsed = 0;
            loopLock = new object();

            loopTimer = new Timer();
            loopTimer.Interval = 1000 / 30;
            loopTimer.Elapsed += Tick;
            //loopTimer.AutoReset = false;
        }

        public void Start()
        {
            lastSignal = DateTime.Now;
            loopTimer.Start();
            display.Start();
        }

        private void Tick(object sender, ElapsedEventArgs e)
        {
            TimeSpan delta = e.SignalTime - lastSignal;
            lastSignal = e.SignalTime;
            Console.WriteLine(delta.Milliseconds);

            try
            {
                loopTimer.Stop();

                lock (loopLock)
                {
                    DoTick();
                }
            }
            finally
            {
                loopTimer.Start();
            }
        }

        private void DoTick()
        {
            Mat frame = capture.SingleCapture();

            detector.InvalidateGrayFrame(frame);
            RotatedRect screen = detector.GetScreen();
            PointF ballCenter = detector.GetBallCenter();
            PointF basketCenter = detector.GetBasket();

            Helpers.DebugInfo("shit");

            elapsed += LoopInterval;

            if (elapsed > 1500) // 1.5 seconds
            {
                // get a prediction, do a throw/swipe, and then get the points
                IntervalAction();
            }

            Image<Bgr, Byte> drawFrame = frame.ToImage<Bgr, Byte>();
            //detector.DrawDebugInfo(drawFrame);
            //display.Display(drawFrame);
            DrawDebugInfo(drawFrame);

            Helpers.AddDebugInfoAndDraw(windowTitle, drawFrame);
        }

        private void IntervalAction()
        {

        }

        public void DrawDebugInfo(Image<Bgr, Byte> image)
        {
            // screen debug info
            if (detector.FoundScreen)
            {
                string rectangleText = String.Format("Screen rect: ({0}, {1}), ({2}, {3})",
                    detector.ScreenBounds.MinAreaRect().X.ToString(),
                    detector.ScreenBounds.MinAreaRect().Y.ToString(),
                    ((int)(detector.ScreenBounds.MinAreaRect().X + detector.ScreenBounds.Size.Width)).ToString(),
                    ((int)(detector.ScreenBounds.MinAreaRect().Y + detector.ScreenBounds.Size.Height)).ToString());

                Helpers.DebugInfo(rectangleText);

                image.Draw(detector.ScreenBounds, new Bgr(Color.Aqua), 3);
            }

            // ball debug info
            if (detector.FoundBall)
            {
                Point pixPos = Helpers.ApproxBallToPixelPos(detector.ScreenBounds, detector.GetBallCenter(), ScreenResWidth, ScreenResHeight);
                Helpers.DebugInfo("Circle pix-pos: ({0}, {1})", pixPos.X, pixPos.Y);

                CvInvoke.Circle(image, detector.GetBallCenter().ToPoint(), 5, new Bgr(255, 0, 0).MCvScalar, 3, LineType.FourConnected); // draw center point
                image.Draw(detector.GetBall(), new Bgr(Color.Blue), 2); // draw circle
            }
        }
    }
}
