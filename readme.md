# trackinig 
- CvFGDetector: foreground extraction
- CvBlobDetector: component blob
```
/* Blob detector creator (sole interface function for this file): */
CvBlobDetector* cvCreateBlobDetectorCC(){return new CvBlobDetectorCC;}

/* Blob detector creator (sole interface function for this file) */
CvBlobDetector* cvCreateBlobDetectorSimple(){return new CvBlobDetectorSimple;}
```
- CvBlobTracker:
- CvBlobTrackGen: trajectory generate
- CvBlobTrackPostProc: trajectory post-processing
- CvBlobTrackAnalysis:

