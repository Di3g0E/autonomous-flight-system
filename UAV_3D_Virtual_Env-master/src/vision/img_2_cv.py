import numpy as np
import cv2 as cv
from panda3d.core import Texture, GraphicsOutput

class opencv_camera():
    def __init__(self, render, name, frame_interval):
        self.frame_int = frame_interval
        self.render = render   
        window_size = (self.render.win.getXSize(), self.render.win.getYSize())     
        self.buffer = self.render.win.makeTextureBuffer(name, *window_size, None, True)
        self.cam = self.render.makeCamera(self.buffer)
        self.cam.setName(name)     
        self.cam.node().getLens().setFilmSize(36, 24)
        self.cam.node().getLens().setFocalLength(45)
        self.name = name
        self.render.taskMgr.add(self.set_active, name) 
        self.render.taskMgr.add(self.set_active, name)
        self.buffer.setActive(0)
        
        # Register depth render texture so we can read it back to RAM
        self.depth_texture = Texture('depth_' + name)
        self.buffer.addRenderTexture(
            self.depth_texture,
            GraphicsOutput.RTMCopyRam,
            GraphicsOutput.RTPDepth
        )
        
        # Store lens parameters for depth conversion
        lens = self.cam.node().getLens()
        self.near_plane = lens.getNear()
        self.far_plane = lens.getFar()
        
    def get_image(self, target_frame=True):
        tex = self.buffer.getTexture()  
        img = tex.getRamImage()
        image = np.frombuffer(img, np.uint8)
        if len(image) > 0:
            image = np.reshape(image, (tex.getYSize(), tex.getXSize(), 4))
            image = cv.resize(image, (0,0), fx=0.5, fy=0.5)
            image = cv.flip(image, 0)
            return True, image
        else:
            return False, None
    
    def get_depth(self, normalize=True, metric=False):
        """
        Extract depth buffer from camera.
        
        Args:
            normalize: Return values in [0, 1] range (default: True)
            metric: Convert to metric depth in meters (default: False)
                   If True, overrides normalize
        
        Returns:
            (success, depth_image): 
                - success: bool indicating if capture succeeded
                - depth_image: np.array of shape (H, W, 1), dtype float32
        """
        # Use the pre-registered depth texture
        depth_tex = self.depth_texture
        
        if depth_tex is None:
            return False, None
        
        # Extract raw depth data
        depth_data = depth_tex.getRamImage()
        
        if depth_data is None or len(depth_data) == 0:
            return False, None
        
        # Convert to numpy array
        # Panda3D depth buffer is typically stored as float32
        depth_array = np.frombuffer(depth_data, np.float32)
        
        # Reshape to image dimensions
        height = depth_tex.getYSize()
        width = depth_tex.getXSize()
        
        try:
            depth_image = np.reshape(depth_array, (height, width))
        except ValueError:
            # If reshape fails, depth buffer might be in different format
            # Try as uint8 and convert
            depth_array = np.frombuffer(depth_data, np.uint8)
            depth_image = np.reshape(depth_array, (height, width))
            depth_image = depth_image.astype(np.float32) / 255.0
        
        # Flip vertically to match OpenCV convention (origin at top-left)
        depth_image = cv.flip(depth_image, 0)
        
        # Resize to match RGB output (0.5x scale)
        depth_image = cv.resize(depth_image, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST)
        
        if metric:
            # Convert from normalized depth buffer to metric depth
            depth_image = self._depth_buffer_to_metric(depth_image)
        elif not normalize:
            # Keep raw values
            pass
        # else: depth_image is already normalized [0, 1]
        
        # Add channel dimension for consistency with image format (H, W, 1)
        depth_image = np.expand_dims(depth_image, axis=-1)
        
        return True, depth_image
    
    def _depth_buffer_to_metric(self, depth_normalized):
        """
        Convert normalized depth buffer [0, 1] to metric depth in meters.
        
        Panda3D uses standard Z-buffer (non-linear) encoding:
        depth_buffer = (far * (z - near)) / (z * (far - near))
        
        Inverse formula to get z (metric depth):
        z = (2 * near * far) / (far + near - depth_ndc * (far - near))
        where depth_ndc = 2 * depth_buffer - 1  (convert [0,1] to [-1,1])
        
        Args:
            depth_normalized: Normalized depth values in [0, 1]
        
        Returns:
            depth_metric: Depth in meters
        """
        near = self.near_plane
        far = self.far_plane
        
        # Convert [0, 1] to NDC space [-1, 1]
        depth_ndc = 2.0 * depth_normalized - 1.0
        
        # Apply inverse perspective projection
        depth_metric = (2.0 * near * far) / (far + near - depth_ndc * (far - near))
        
        # Clamp to valid range (avoid numerical issues)
        depth_metric = np.clip(depth_metric, near, far)
        
        return depth_metric
    
    def set_active(self, task):
        if task.frame % 10 == 0:
            self.buffer.setActive(1)
        return task.cont
    
    def set_inactive(self, task):
        if task.frame % 10 == 1:
            self.buffer.setActive(0)
        return task.cont