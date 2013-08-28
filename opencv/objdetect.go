package opencv

//#include "opencv.h"
//#include <opencv2/objdetect/objdetect.hpp>
//#cgo linux  pkg-config: opencv
//#cgo darwin pkg-config: opencv
//#cgo windows LDFLAGS: -lopencv_core242.dll -lopencv_imgproc242.dll -lopencv_photo242.dll -lopencv_highgui242.dll -lstdc++
import "C"
import "unsafe"

type HaarClassifierCascade C.CvHaarClassifierCascade

func LoadClassifier(filename string) *HaarClassifierCascade {
	name_c := C.CString(filename)
	defer C.free(unsafe.Pointer(name_c))

	return (*HaarClassifierCascade)(C.cvLoad(name_c, nil, nil, nil))
}

func (cascade *HaarClassifierCascade) Release() {
	cascade_c := (*C.CvHaarClassifierCascade)(cascade)
	C.cvReleaseHaarClassifierCascade(&cascade_c)
}

/*
CVAPI(CvSeq*) cvHaarDetectObjects( const CvArr* image,
                     CvHaarClassifierCascade* cascade, CvMemStorage* storage,
                     double scale_factor CV_DEFAULT(1.1),
                     int min_neighbors CV_DEFAULT(3), int flags CV_DEFAULT(0),
                     CvSize min_size CV_DEFAULT(cvSize(0,0)), CvSize max_size CV_DEFAULT(cvSize(0,0)));
*/
func (cascade *HaarClassifierCascade) DetectObjects(image *IplImage, storage *MemStorage, scale_factor float64, min_neighbors int, flags int, min_size Size, max_size Size) *Seq {
	min_size_c := C.cvSize(C.int(min_size.Width), C.int(min_size.Height))
	max_size_c := C.cvSize(C.int(max_size.Width), C.int(max_size.Height))
	return (*Seq)(C.cvHaarDetectObjects(
		unsafe.Pointer(image),
		(*C.CvHaarClassifierCascade)(cascade),
		(*C.CvMemStorage)(storage),
		C.double(scale_factor),
		C.int(min_neighbors),
		C.int(flags),
		min_size_c,
		max_size_c,
	))
}
