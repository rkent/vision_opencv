import cv2
import math
import copy
import numpy
import numpy as np
import warnings

def mkmat(rows, cols, L) -> numpy.ndarray:
    # mat = numpy.matrix(L, dtype='float64')
    mat = np.array(L,dtype='float64')
    mat.resize(rows,cols)
    return mat

class PinholeCameraModel:

    """
    A pinhole camera is an idealized monocular camera.
    """

    def __init__(self):
        self._k = None
        self._d = None
        self._r = None
        self._p = None
        self._full_K = None
        self._full_P = None
        self._width = None
        self._height = None
        self._binning_x = None
        self._binning_y = None
        self._raw_roi = None
        self._tf_frame = None
        self._stamp = None
        self._resolution = None

    def fromCameraInfo(self, msg):
        """
        :param msg: camera parameters
        :type msg:  sensor_msgs.msg.CameraInfo

        Set the camera parameters from the :class:`sensor_msgs.msg.CameraInfo` message.
        """
        self._k = mkmat(3, 3, msg.k)
        if msg.d:
            self._d = mkmat(len(msg.d), 1, msg.d)
        else:
            self._d = None
        self._r = mkmat(3, 3, msg.r)
        self._p = mkmat(3, 4, msg.p)
        self._full_K = mkmat(3, 3, msg.k)
        self._full_P = mkmat(3, 4, msg.p)
        self._width = msg.width
        self._height = msg.height
        self._binning_x = max(1, msg.binning_x)
        self._binning_y = max(1, msg.binning_y)
        self._resolution = (msg.width, msg.height)

        self._raw_roi = copy.copy(msg.roi)
        # ROI all zeros is considered the same as full resolution
        if (self._raw_roi.x_offset == 0 and self._raw_roi.y_offset == 0 and
            self._raw_roi.width == 0 and self._raw_roi.height == 0):
            self._raw_roi.width = self._width
            self._raw_roi.height = self._height
        self._tf_frame = msg.header.frame_id
        self._stamp = msg.header.stamp

        # Adjust K and P for binning and ROI
        self._k[0,0] /= self._binning_x
        self._k[1,1] /= self._binning_y
        self._k[0,2] = (self._k[0,2] - self._raw_roi.x_offset) / self._binning_x
        self._k[1,2] = (self._k[1,2] - self._raw_roi.y_offset) / self._binning_y
        self._p[0,0] /= self._binning_x
        self._p[1,1] /= self._binning_y
        self._p[0,2] = (self._p[0,2] - self._raw_roi.x_offset) / self._binning_x
        self._p[1,2] = (self._p[1,2] - self._raw_roi.y_offset) / self._binning_y

    def rectifyImage(self, raw, rectified):
        """
        :param raw:       input image
        :type raw:        :class:`CvMat` or :class:`IplImage`
        :param rectified: rectified output image
        :type rectified:  :class:`CvMat` or :class:`IplImage`

        Applies the rectification specified by camera parameters :math:`K` and and :math:`D` to image `raw` and writes the resulting image `rectified`.
        """

        self.mapx = numpy.ndarray(shape=(self._height, self._width, 1),
                           dtype='float32')
        self.mapy = numpy.ndarray(shape=(self._height, self._width, 1),
                           dtype='float32')
        cv2.initUndistortRectifyMap(self._k, self._d, self._r, self._p,
                (self._width, self._height), cv2.CV_32FC1, self.mapx, self.mapy)
        cv2.remap(raw, self.mapx, self.mapy, cv2.INTER_CUBIC, rectified)

    def rectifyPoint(self, uv_raw):
        """
        :param uv_raw:    pixel coordinates
        :type uv_raw:     (u, v)

        Applies the rectification specified by camera parameters
        :math:`K` and and :math:`D` to point (u, v) and returns the
        pixel coordinates of the rectified point.
        """

        src = mkmat(1, 2, list(uv_raw))
        src.resize((1,1,2))
        dst = cv2.undistortPoints(src, self._k, self._d, R=self._r, P=self._p)
        return dst[0,0]

    def project3dToPixel(self, point):
        """
        :param point:     3D point
        :type point:      (x, y, z)

        Returns the rectified pixel coordinates (u, v) of the 3D point,
        using the camera :math:`P` matrix.
        This is the inverse of :math:`projectPixelTo3dRay`.
        """
        src = mkmat(4, 1, [point[0], point[1], point[2], 1.0])
        dst = self._p @ src
        x = dst[0,0]
        y = dst[1,0]
        w = dst[2,0]
        if w != 0:
            return (x / w, y / w)
        else:
            return (float('nan'), float('nan'))

    def projectPixelTo3dRay(self, uv):
        """
        :param uv:        rectified pixel coordinates
        :type uv:         (u, v)

        Returns the unit vector which passes from the camera center to through rectified pixel (u, v),
        using the camera :math:`P` matrix.
        This is the inverse of :math:`project3dToPixel`.
        """
        x = (uv[0] - self.cx()) / self.fx()
        y = (uv[1] - self.cy()) / self.fy()
        norm = math.sqrt(x*x + y*y + 1)
        x /= norm
        y /= norm
        z = 1.0 / norm
        return (x, y, z)

    def getDeltaU(self, deltaX, Z):
        """
        :param deltaX:          delta X, in cartesian space
        :type deltaX:           float
        :param Z:               Z, in cartesian space
        :type Z:                float
        :rtype:                 float

        Compute delta u, given Z and delta X in Cartesian space.
        For given Z, this is the inverse of :math:`getDeltaX`.
        """
        if Z == 0:
            return float('inf')
        else:
            return self.fx() * deltaX / Z

    def getDeltaV(self, deltaY, Z):
        """
        :param deltaY:          delta Y, in cartesian space
        :type deltaY:           float
        :param Z:               Z, in cartesian space
        :type Z:                float
        :rtype:                 float

        Compute delta v, given Z and delta Y in Cartesian space.
        For given Z, this is the inverse of :math:`getDeltaY`.
        """
        if Z == 0:
            return float('inf')
        else:
            return self.fy() * deltaY / Z

    def getDeltaX(self, deltaU, Z):
        """
        :param deltaU:          delta u in pixels
        :type deltaU:           float
        :param Z:               Z, in cartesian space
        :type Z:                float
        :rtype:                 float

        Compute delta X, given Z in cartesian space and delta u in pixels.
        For given Z, this is the inverse of :math:`getDeltaU`.
        """
        return Z * deltaU / self.fx()

    def getDeltaY(self, deltaV, Z):
        """
        :param deltaV:          delta v in pixels
        :type deltaV:           float
        :param Z:               Z, in cartesian space
        :type Z:                float
        :rtype:                 float

        Compute delta Y, given Z in cartesian space and delta v in pixels.
        For given Z, this is the inverse of :math:`getDeltaV`.
        """
        return Z * deltaV / self.fy()

    def fullResolution(self)->tuple[int, int]:
        """
        :rtype:                 tuple[int, int]

        Returns the full resolution of the camera as a tuple in the format (width, height)
        """
        return self._resolution
    
    def intrinsicMatrix(self)->numpy.matrix:
        """ 
        :rtype:                 numpy.matrix

        PinholeCameraModel.intrinsicMatrix()->numpy.matrix is depricated. Please use intrinsic_matrix()->numpy.ndarray
        Returns :math:`K`, also called camera_matrix in cv docs 
        """
        warnings.warn("PinholeCameraModel.intrinsicMatrix()->numpy.matrix is depricated. Please use intrinsic_matrix()->numpy.ndarray", DeprecationWarning)
        return self._k

    def distortionCoeffs(self)->numpy.matrix:
        """ 
        :rtype:                 numpy.matrix
        
        PinholeCameraModel.distortionCoeffs()->numpy.matrix is depricated. Please use distortion_coeffs()->numpy.ndarray
        Returns :math:`D` 
        """
        warnings.warn("PinholeCameraModel.distortionCoeffs()->numpy.matrix is depricated. Please use distortion_coeffs()->numpy.ndarray", DeprecationWarning)
        return self._d

    def rotationMatrix(self)->numpy.matrix:
        """ 
        :rtype:                 numpy.matrix

        PinholeCameraModel.rotationMatrix()->numpy.matrix is depricated. Please use rotation_matrix()->numpy.ndarray
        Returns :math:`R` 
        """
        warnings.warn("PinholeCameraModel.rotationMatrix()->numpy.matrix is depricated. Please use rotation_matrix()->numpy.ndarray", DeprecationWarning)
        return self._r

    def projection_matrix(self) ->numpy.ndarray:
        """ 
        :rtype:                 numpy.ndarray

        Returns :math:`P` 
        """
        return self._p
    
    def projectionMatrix(self) -> numpy.matrix:
        """ 
        :rtype:                 numpy.matrix

        PinholeCameraModel.projectionMatrix()->numpy.matrix is depricated. Please use projection_matrix()->numpy.ndarray
        Returns :math:`P` 
        """
        warnings.warn("PinholeCameraModel.projectionMatrix()->numpy.matrix is depricated. Please use projection_matrix()->numpy.ndarray", DeprecationWarning)
        return np.matrix(self.projection_matrix(), dtype='float64')


    def fullIntrinsicMatrix(self) -> numpy.matrix:
        """ 
        :rtype:                 numpy.matrix

        PinholeCameraModel.fullIntrinsicMatrix()->numpy.matrix is depricated. Please use full_intrinsic_matrix()->numpy.ndarray"
        Return the original camera matrix for full resolution 
        """
        warnings.warn("PinholeCameraModel.fullIntrinsicMatrix()->numpy.matrix is depricated. Please use full_intrinsic_matrix()->numpy.ndarray", DeprecationWarning)        
        return self._full_K

    def fullProjectionMatrix(self)->numpy.matrix:
        """ 
        :rtype:                 numpy.matrix

        PinholeCameraModel.fullProjectionMatrix()->numpy.matrix is depricated. Please use full_projection_matrix()->numpy.ndarray
        Return the projection matrix for full resolution """
        warnings.warn("PinholeCameraModel.fullProjectionMatrix()->numpy.matrix is depricated. Please use full_projection_matrix()->numpy.ndarray", DeprecationWarning)        
        return self._full_P

    def cx(self)->float:
        """ 
        :rtype:                 float      
        
        Returns x center """
        return self._p[0,2]

    def cy(self)->float:
        """ 
        :rtype:                 float      
        
        Returns y center 
        """
        return self._p[1,2]

    def fx(self)->float:
        """ 
        :rtype:                 float      
        
        Returns x focal length 
        """
        return self._p[0,0]

    def fy(self)->float:
        """ 
        :rtype:                 float      
        
        Returns y focal length 
        """
        return self._p[1,1]

    def Tx(self)->float:
        """ 
        :rtype:                 float      
        
        Return the x-translation term of the projection matrix 
        """
        return self._p[0,3]

    def Ty(self)->float:
        """ 
        :rtype:                 float      
        
        Return the y-translation term of the projection matrix 
        """
        return self._p[1,3]

    def fovX(self)->float:
        """ 
        :rtype:                 float      
        
        Returns the horizontal field of view in radians.
        Horizontal FoV = 2 * arctan((width) / (2 * Horizontal Focal Length) )
        """
        return 2 * math.atan(self._width / (2 * self.fx()))

    def fovY(self)->float:
        """ 
        :rtype:                 float      
        
        Returns the vertical field of view in radians.
        Vertical FoV = 2 * arctan((height) / (2 * Vertical Focal Length) )
        """
        return 2 * math.atan(self._height / (2 * self.fy()))

    def tfFrame(self)->str:
        """ 
        :rtype:                 str      
        
        Returns the tf frame name - a string - of the camera.
        This is the frame of the :class:`sensor_msgs.msg.CameraInfo` message.
        """
        return self._tf_frame

class StereoCameraModel:
    """
    An idealized stereo camera.
    """
    def __init__(self):
        self._left = PinholeCameraModel()
        self._right = PinholeCameraModel()
        self._q = None

    def fromCameraInfo(self, left_msg, right_msg):
        """
        :param left_msg: left camera parameters
        :type left_msg:  sensor_msgs.msg.CameraInfo
        :param right_msg: right camera parameters
        :type right_msg:  sensor_msgs.msg.CameraInfo

        Set the camera parameters from the :class:`sensor_msgs.msg.CameraInfo` messages.
        """
        self._left.fromCameraInfo(left_msg)
        self._right.fromCameraInfo(right_msg)

        # [ Fx, 0,  Cx,  Fx*-Tx ]
        # [ 0,  Fy, Cy,  0      ]
        # [ 0,  0,  1,   0      ]

        assert self._right._p is not None
        fx = self._right.projection_matrix()[0, 0]
        cx = self._right.projection_matrix()[0, 2]
        cy = self._right.projection_matrix()[1, 2]
        tx = -self._right.projection_matrix()[0, 3] / fx

        # Q is:
        #    [ 1, 0,  0, -Clx ]
        #    [ 0, 1,  0, -Cy ]
        #    [ 0, 0,  0,  Fx ]
        #    [ 0, 0, 1 / Tx, (Crx-Clx)/Tx ]

        self._q = numpy.zeros((4, 4), dtype='float64')
        self._q[0, 0] = 1.0
        self._q[0, 3] = -cx
        self._q[1, 1] = 1.0
        self._q[1, 3] = -cy
        self._q[2, 3] = fx
        self._q[3, 2] = 1 / tx

    def tfFrame(self)->str:
        """ 
        :rtype:                 str      
        
        Returns the tf frame name - a string - of the camera.
        This is the frame of the :class:`sensor_msgs.msg.CameraInfo` message.
        """

        return self._left.tfFrame()

    def project3dToPixel(self, point)->tuple[tuple[float,float],tuple[float,float]]:
        """
        :param point:     3D point
        :type point:      (x, y, z)

        Returns the rectified pixel coordinates (u, v) of the 3D point, for each camera, as ((u_left, v_left), (u_right, v_right))
        using the cameras' :math:`P` matrices.
        This is the inverse of :math:`projectPixelTo3d`.
        """
        l = self._left.project3dToPixel(point)
        r = self._right.project3dToPixel(point)
        return (l, r)

    def projectPixelTo3d(self, left_uv, disparity):
        """
        :param left_uv:        rectified pixel coordinates
        :type left_uv:         (u, v)
        :param disparity:        disparity, in pixels
        :type disparity:         float

        Returns the 3D point (x, y, z) for the given pixel position,
        using the cameras' :math:`P` matrices.
        This is the inverse of :math:`project3dToPixel`.

        Note that a disparity of zero implies that the 3D point is at infinity.
        """
        src = mkmat(4, 1, [left_uv[0], left_uv[1], disparity, 1.0])
        dst = self._q @ src
        x = dst[0, 0]
        y = dst[1, 0]
        z = dst[2, 0]
        w = dst[3, 0]
        if w != 0:
            return (x / w, y / w, z / w)
        else:
            return (0.0, 0.0, 0.0)

    def getZ(self, disparity):
        """
        :param disparity:        disparity, in pixels
        :type disparity:         float

        Returns the depth at which a point is observed with a given disparity.
        This is the inverse of :math:`getDisparity`.

        Note that a disparity of zero implies Z is infinite.
        """
        if disparity == 0:
            return float('inf')
        Tx = -self._right.projectionMatrix()[0, 3]
        return Tx / disparity

    def getDisparity(self, Z):
        """
        :param Z:          Z (depth), in cartesian space
        :type Z:           float

        Returns the disparity observed for a point at depth Z.
        This is the inverse of :math:`getZ`.
        """
        if Z == 0:
            return float('inf')
        Tx = -self._right.projection_matrix()[0, 3]
        return Tx / Z
    
    
    def get_left_camera(self)->PinholeCameraModel:
        """ 
        Returns the PinholeCameraModel object of the left camera
        """
        return self._left
    
    def get_right_camera(self)->PinholeCameraModel:
        """ 
        Returns the PinholeCameraModel object of the right camera
        """
        return self._right
    
if __name__ == '__main__':
#    from __future__ import print_function

    import unittest
    import sensor_msgs.msg

    #from image_geometry import PinholeCameraModel, StereoCameraModel

    class TestDirected(unittest.TestCase):

        def setUp(self):
            pass

        def test_monocular(self):
            ci = sensor_msgs.msg.CameraInfo()
            ci.width = 640
            ci.height = 480
            print(ci)
            cam = PinholeCameraModel()
            cam.fromCameraInfo(ci)
            print(cam.rectifyPoint((0, 0)))

            print(cam.project3dToPixel((0,0,0)))

        def test_stereo(self):
            lmsg = sensor_msgs.msg.CameraInfo()
            rmsg = sensor_msgs.msg.CameraInfo()
            for m in (lmsg, rmsg):
                m.width = 640
                m.height = 480

            # These parameters taken from a real camera calibration
            lmsg.d =  [-0.363528858080088, 0.16117037733986861, -8.1109585007538829e-05, -0.00044776712298447841, 0.0]
            lmsg.k =  [430.15433020105519, 0.0, 311.71339830549732, 0.0, 430.60920415473657, 221.06824942698509, 0.0, 0.0, 1.0]
            lmsg.r =  [0.99806560714807102, 0.0068562422224214027, 0.061790256276695904, -0.0067522959054715113, 0.99997541519165112, -0.0018909025066874664, -0.061801701660692349, 0.0014700186639396652, 0.99808736527268516]
            lmsg.p =  [295.53402059708782, 0.0, 285.55760765075684, 0.0, 0.0, 295.53402059708782, 223.29617881774902, 0.0, 0.0, 0.0, 1.0, 0.0]

            rmsg.d =  [-0.3560641041112021, 0.15647260261553159, -0.00016442960757099968, -0.00093175810713916221]
            rmsg.k =  [428.38163131344191, 0.0, 327.95553847249192, 0.0, 428.85728580588329, 217.54828640915309, 0.0, 0.0, 1.0]
            rmsg.r =  [0.9982082576219119, 0.0067433328293516528, 0.059454199832973849, -0.0068433268864187356, 0.99997549128605434, 0.0014784127772287513, -0.059442773257581252, -0.0018826283666309878, 0.99822993965212292]
            rmsg.p =  [295.53402059708782, 0.0, 285.55760765075684, -26.507895206214123, 0.0, 295.53402059708782, 223.29617881774902, 0.0, 0.0, 0.0, 1.0, 0.0]

            cam = StereoCameraModel()
            cam.fromCameraInfo(lmsg, rmsg)

            for x in (16, 320, m.width - 16):
                for y in (16, 240, m.height - 16):
                    for d in range(1, 10):
                        pt3d = cam.projectPixelTo3d((x, y), d)
                        ((lx, ly), (rx, ry)) = cam.project3dToPixel(pt3d)
                        self.assertAlmostEqual(y, ly, 3)
                        self.assertAlmostEqual(y, ry, 3)
                        self.assertAlmostEqual(x, lx, 3)
                        self.assertAlmostEqual(x, rx + d, 3)

            u = 100.0
            v = 200.0
            du = 17.0
            dv = 23.0
            Z = 2.0
            xyz0 = cam.get_left_camera().projectPixelTo3dRay((u, v))
            xyz0 = (xyz0[0] * (Z / xyz0[2]), xyz0[1] * (Z / xyz0[2]), Z)
            xyz1 = cam.get_left_camera().projectPixelTo3dRay((u + du, v + dv))
            xyz1 = (xyz1[0] * (Z / xyz1[2]), xyz1[1] * (Z / xyz1[2]), Z)
            self.assertAlmostEqual(cam.get_left_camera().getDeltaU(xyz1[0] - xyz0[0], Z), du, 3)
            self.assertAlmostEqual(cam.get_left_camera().getDeltaV(xyz1[1] - xyz0[1], Z), dv, 3)
            self.assertAlmostEqual(cam.get_left_camera().getDeltaX(du, Z), xyz1[0] - xyz0[0], 3)
            self.assertAlmostEqual(cam.get_left_camera().getDeltaY(dv, Z), xyz1[1] - xyz0[1], 3)

    suite = unittest.TestSuite()
    suite.addTest(TestDirected('test_stereo'))
    unittest.TextTestRunner(verbosity=2).run(suite)




