"""Live view from the DWARF's camera, turned into a stream the page can show.

The DWARF 3 streams its telephoto (ch0) and wide (ch1) cameras over RTSP on
port 554, the same feed Dwarfium uses. OpenCV reads it and the frames are
served to the page as an MJPEG stream. One reader runs per camera and stops a
few seconds after the last viewer leaves.
"""
from __future__ import annotations

import os
import threading
import time

from . import config, scope

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


class LiveError(Exception):
    pass


class _Reader:
    def __init__(self, url: str):
        self.url = url
        self.frame: bytes | None = None
        self.error: str | None = None
        self.viewers = 0
        self.last_viewer = time.monotonic()
        self.cv = threading.Condition()
        self.alive = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            import cv2
        except ImportError:
            self.error = "Live view needs opencv; run the launcher again to install it."
            self.alive = False
            return
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            self.error = "The DWARF's live stream didn't open. Open the camera in Dwarfium once, then try again."
            self.alive = False
            with self.cv:
                self.cv.notify_all()
            return
        fails = 0
        while self.alive:
            ok, img = cap.read()
            if not ok:
                fails += 1
                if fails > 50:
                    self.error = "The live stream stopped."
                    break
                time.sleep(0.05)
                continue
            fails = 0
            h, w = img.shape[:2]
            if w > 1280:
                img = cv2.resize(img, (1280, int(h * 1280 / w)), interpolation=cv2.INTER_AREA)
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self.cv:
                    self.frame = jpg.tobytes()
                    self.cv.notify_all()
            if self.viewers == 0 and time.monotonic() - self.last_viewer > 8:
                break
        cap.release()
        self.alive = False
        with self.cv:
            self.cv.notify_all()


_READERS: dict[str, _Reader] = {}
_LOCK = threading.Lock()


def _reader(camera: str) -> _Reader:
    ip = (config.load().get("dwarf_ip") or "").strip()
    if not scope._is_local(ip):
        raise LiveError("Find your DWARF first.")
    url = f"rtsp://{ip}:554/ch{0 if camera == 'tele' else 1}/stream0"
    with _LOCK:
        r = _READERS.get(url)
        if not r or not r.alive:
            r = _READERS[url] = _Reader(url)
        return r


def frames(camera: str = "tele"):
    """Yield JPEG frames for one viewer until the stream ends."""
    r = _reader(camera)
    r.viewers += 1
    try:
        last = None
        while True:
            with r.cv:
                r.cv.wait_for(lambda: r.frame is not last or not r.alive, timeout=10)
            if not r.alive and r.frame is last:
                raise LiveError(r.error or "The live stream stopped.")
            if r.frame is last:
                raise LiveError("No picture from the DWARF for 10 seconds.")
            last = r.frame
            yield last
    finally:
        r.viewers -= 1
        r.last_viewer = time.monotonic()
