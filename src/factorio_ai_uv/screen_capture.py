"""
Factorio screen capture using mss.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import mss
import numpy as np
from PIL import Image

from .config import CaptureConfig
from .exceptions import ScreenCaptureError

logger = logging.getLogger(__name__)


@dataclass
class CaptureRegion:
    """Screen capture region."""
    top: int
    left: int
    width: int
    height: int


class FactorioScreenCapture:
    """
    Captures Factorio game window and preprocesses for CNN.
    
    Features:
    - Auto-detects Factorio window
    - Configurable capture region and interval
    - Resizes to CNN input size (84x84)
    - Normalizes to [0, 1] float32 CHW format
    """
    
    def __init__(
        self,
        config: Optional["CaptureConfig"] = None,
    ) -> None:
        """
        Initialize screen capture.
        
        Args:
            config: Capture configuration
        """
        self.config = config or CaptureConfig()
        self.sct = mss.mss()
        self._last_frame = np.zeros(
            (self.config.size[1], self.config.size[0], 3), 
            dtype=np.uint8
        )
        self._tick_counter = 0
        self._last_capture_tick = 0
        self._box: dict = {}
        
        self._detect_window()

    def _detect_window(self) -> None:
        """Auto-detect Factorio window."""
        try:
            import pygetwindow as gw
            windows = gw.getAllWindows()
            factorio_windows = [
                w for w in windows 
                if "factorio" in w.title.lower()
            ]
            if factorio_windows:
                win = factorio_windows[0]
                if win.isMinimized:
                    win.restore()
                self._box = {
                    "top": win.top,
                    "left": win.left,
                    "width": win.width,
                    "height": win.height,
                }
                logger.info(f"Auto-detected Factorio window: {self._box}")
                return
        except ImportError:
            logger.warning("pygetwindow not available, using fallback region")
        except Exception as e:
            logger.warning(f"Window detection failed: {e}")

        # Fallback
        self._box = {
            "top": 40, "left": 100, 
            "width": 800, "height": 600
        }
        logger.info(f"Using fallback capture region: {self._box}")

    def set_region(self, region: dict) -> None:
        """Set capture region manually."""
        self._box = region
        logger.info(f"Set capture region: {region}")

    def capture(self) -> np.ndarray:
        """
        Capture and preprocess frame.
        
        Returns:
            Preprocessed frame (H, W, C) uint8 [0, 255]
        """
        self._tick_counter += 1
        
        # Skip captures based on interval
        if self._tick_counter - self._last_capture_tick < self.config.interval:
            return self._last_frame

        try:
            grab = self.sct.grab(self._box)
            img = Image.frombytes("RGB", grab.size, grab.rgb)
            img = img.resize(self.config.size, Image.LANCZOS)
            arr = np.array(img, dtype=np.uint8)  # HWC
            
            self._last_frame = arr
            self._last_capture_tick = self._tick_counter
            
            return arr
            
        except Exception as e:
            logger.error(f"Screen capture error: {e}")
            return self._last_frame

    def get_region(self) -> dict:
        """Get current capture region."""
        return self._box.copy()

    def close(self) -> None:
        """Cleanup."""
        self.sct.close()


# Test
if __name__ == "__main__":
    from .config import CaptureConfig
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    config = CaptureConfig()
    capture = FactorioScreenCapture(config)
    
    print("Testing screen capture...")
    for i in range(5):
        frame = capture.capture()
        print(f"Capture {i+1}: shape={frame.shape}, dtype={frame.dtype}, "
              f"min={frame.min():.3f}, max={frame.max():.3f}")
        time.sleep(0.5)
    
    print("Screen capture working!")