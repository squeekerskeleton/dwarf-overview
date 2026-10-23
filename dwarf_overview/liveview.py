"""Live view from the DWARF's camera, turned into a stream the page can show.

The DWARF 3 streams its telephoto (ch0) and wide (ch1) cameras over RTSP on
port 554, the same feed Dwarfium uses. The video only flows once the camera is
switched on, so the app does that first (control.open_preview), as Dwarfium's
Camera page does. OpenCV reads it and the frames are
served to the page as an MJPEG stream. One reader runs per camera and stops a
few seconds after the last viewer leaves.
"""
from __future__ import annotations

import os
import threading
import time

from . import config, scope

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
OPEN_TRIES = 4


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
        cap = None
        for attempt in range(OPEN_TRIES):  # the stream takes a moment to start after the camera opens
            params = [getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", 53), 4000, getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", 54), 5000]
            try:
                cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, params)
            except TypeError:  # older OpenCV without open parameters
                cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if cap.isOpened():
                break
            cap.release()
            cap = None
            time.sleep(1)
        if cap is None:
            self.error = (f"The camera is on but its video stream didn't start after {OPEN_TRIES} tries. "
                          "Another app may be using the camera; close it and click Show live view again.")
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


def _reader(camera: str, go_live: bool = False) -> _Reader:
    from . import control
    ip = (config.load().get("dwarf_ip") or "").strip()
    if not scope._is_local(ip):
        raise LiveError("Find your DWARF first (Network tab or Plan & Control).")
    url = f"rtsp://{ip}:554/ch{0 if camera == 'tele' else 1}/stream0"
    with _LOCK:
        r = _READERS.get(url)
        if r and r.alive:
            return r
    # Turn the camera's video on first, so there's no need to open Dwarfium's Camera page.
    try:
        opened = control.open_preview(camera, go_live=go_live)
    except control.ControlError as e:
        raise LiveError(f"Couldn't switch the DWARF's camera on: {e}")
    time.sleep(2)  # give the camera a moment to start streaming
    with _LOCK:
        r = _READERS.get(url)
        if not r or not r.alive:
            r = _READERS[url] = _Reader(url)
        r.opened = opened
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


def start(camera: str = "tele", timeout: float = 30) -> dict:
    """Switch the camera on and wait for the first picture, so the page can show a clear error if it fails.

    If the plain way gives no picture, try once more with goLive, which a cold DWARF needs."""
    for go_live in (False, True):
        r = _reader(camera, go_live)
        with r.cv:
            r.cv.wait_for(lambda: r.frame is not None or not r.alive, timeout=timeout)
        if r.frame is not None:
            break
        r.alive = False
    if r.frame is None:
        o = getattr(r, "opened", {}) or {}
        codes = f"enterCamera {o.get('enter_code')}, goLive {o.get('live_code')}, preview {o.get('preview_code')}"
        raise LiveError(f"{r.error or 'The camera is on but no picture arrived.'} (DWARF replies: {codes}; "
                        "0 means OK.) If the DWARFLAB phone app or Dwarfium is open, close it and try again.")
    r.last_viewer = time.monotonic()  # keep it running while the page connects
    return {"ok": True, "camera": camera}
