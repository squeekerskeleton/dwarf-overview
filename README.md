# DWARF Post-Processing Hub

A small app for your laptop that looks at a DWARF 3 image, measures what needs fixing, asks Claude for a processing plan, and drives GraXpert and Siril for you. Every session is logged, and your ratings feed into the next suggestions.

## Set up on Windows (once)

1. **Python.** Install Python 3.10 or newer from https://www.python.org/downloads/. On the first installer screen, tick **"Add python.exe to PATH"**.
2. **GraXpert and Siril** (free). GraXpert: https://graxpert.com. Siril 1.2 or newer: https://siril.org. Use the default install locations and the app finds them on its own.
3. **Anthropic API key.** Create one at https://console.anthropic.com (Settings → API Keys) and add a few dollars of credit. Each assessment costs roughly 10 to 15 cents.
4. Unzip this folder somewhere permanent, such as `Documents\DWARF Overview`.
5. Double-click **`start-windows.bat`**. The first run takes a minute to set up, then your browser opens the app. Keep the black window open while you use it, and close it to stop.
6. Click **Settings**, paste your API key, and save. The key is stored only on your laptop.
7. **ASTAP** (free, optional but recommended). It names the subject by matching the stars in your photo. Install it from https://www.hnsky.org/astap.htm, plus the **D50** star database from the same page. Use the default install location and the app finds it.

Windows may warn about the `.bat` file the first time. Choose **More info → Run anyway**.

**Setting up another PC?** The first time the app opens, a **Setup** page lists every program it uses, whether each is installed, and a **Download** button for each one's official site. Install them, then click **Check again**. You can reopen it anytime with the **Setup** button. Each time the Windows launcher runs, it does a quick software check first and stops only if Python or its libraries are missing.

## Using it

1. Plug the DWARF 3 in over USB-C (or copy its files to the laptop). In the left panel, open `Astronomy → DWARF_RAW_<target>_…` and click the **stacked FITS** file. The FITS works much better than the JPG because it hasn't been stretched yet.
2. The app measures the image right away, on your laptop and for free:
   - **Gradient**: light pollution or moon glow across the frame
   - **Signal/noise**
   - **Star count, size (FWHM) and roundness**: shows trailing, wind, focus or dew problems
   - **Colour cast**: a green cast is normal for raw DWARF data
   - **Stretch state and clipping**
3. Optionally add a note ("windy after 1 am", "moon rising"), then click **Assess and plan processing**. Claude returns:
   - A score and the issues it found.
   - A step-by-step plan, with the menu path and exact values for each step, plus buttons that open GraXpert or Siril on the file.
   - **Run GraXpert**: runs background extraction and AI denoise with the suggested settings. You can tweak the numbers first. It writes new files (`…_gx_bg`, `…_gx_bg_dn`) next to the original and never changes your original.
   - **Siril script**: colour fixes and stretch, which you can edit before running. It saves `…_siril.fit` and `…_siril.tif`.
   - Tips for your next night out.
4. Click **Measure the result** after a run to check that the gradient and colour numbers improved.
5. **Rate the result** and note what you actually changed. This is the step that makes it learn: your ratings and notes go along with every future request, with sessions on the same target and filter weighted first.

The **History** tab lists every session, so you can see your star sizes, gradients and scores change over time.

**Night mode** turns the screen red so it doesn't ruin your night vision in the car.

## MOON STACK (4K)

**MOON STACK** at the top lists every night you shot the Moon, found by itself in the library, on the DWARF over USB-C, or over Wi-Fi: the DWARF's raw Moon sessions (`DWARF_RAW_TELE_Moon_...`) and its Moon videos (`Videos`). Press **Stack this night** and it:

1. gathers that night's shots, leaving out any taken more than 45 minutes from the rest (the lit edge and the Moon's tilt change too much to combine);
2. scores every frame's sharpness and keeps the best 10% (at least 30, at most 500), skipping frames where the Moon is cut off by the edge or lost in glare;
3. aligns each kept frame to the sharpest one: rotation and shift for the whole frame (the alt-az mount turns the field), then a grid of surface patches each find their own small shift, the way AutoStakkert undoes seeing ripples; then averages them;
4. sharpens with Richardson-Lucy and a light two-scale sharpen, keeping the limb free of halos, and balances the colour to a neutral grey;
5. centres the Moon in a 3840×2160 image and saves `result_moon_4k` as 16-bit TIFF, PNG, JPG and FITS in `Moon_<date>` inside the processing folder.

Videos and raw sessions from the same night are stacked **together**: 4K videos at full size, HD videos and the raw 1920×1080 sessions at 2×. Each video and session is ranked against itself and gives its best tenth, and every frame's tones are matched to the reference so gamma-encoded video and linear raw frames average cleanly. Each night has tick boxes for its videos and sessions, so you can leave one out; the **Also include a Moon video from elsewhere** box takes a full path to a video or session anywhere else on your PC; **Use** limits the stack to videos or raw frames only. A night of 4K video takes about half an hour; PAUSE ALL and Stop all work as usual.

**From the Assess tab:** when you assess any Moon photo, video or raw frame, an **Auto-stack this night's Moon (4K)** card appears under the results. It shows what that night has and, with one click, stacks the file you assessed together with every other Moon video and raw session from the same night.

The Assess tab picks Moon or deep-sky mode by itself: from the target the DWARF recorded, then the file and folder names (and very short exposures), then the picture (one large bright disc). The reason is shown under the photo, and the Deep sky / Moon buttons switch it.

## Moon mode

The app recognizes Moon images by itself. You can also switch with the **Deep sky / Moon** buttons above the preview.

- **Moon video (SER, MP4, AVI):** it rates the frames one by one and reports:
  - how many frames are close to the sharpest
  - how much the seeing varied
  - whether the Moon drifted or touched the edge
  - whether haze dimmed some frames

  Claude then suggests the AutoStakkert settings: the percentage of frames to stack, the alignment point size, and whether drizzle is worth it. It also says when to run PIPP first.
- **Moon stack (TIF, PNG, FITS):** it measures detail, clipped highlights on the bright limb, and dark halos left by too much sharpening. Claude then suggests RegiStax wavelet values or ImPPG deconvolution settings.
- **Open in PIPP / AutoStakkert / RegiStax / ImPPG:** the app looks for these in the usual install folders, your Desktop and Downloads. If it can't find one, give it the path in Settings → Moon programs. There's also a slot for one more program of your own.
- Moon sessions are logged separately from deep-sky ones, so each kind of advice learns from its own history.

These Moon programs have no command line, so the app opens them with your file loaded and lists the exact values to set.

## What's in the photo, light and colour, and comparisons

As soon as you open an image, the **Subject** card at the top of Measurements names what you're pointed at. ASTAP plate-solves the star field to find the exact patch of sky, and the app looks it up in a built-in catalogue of 2,451 galaxies, nebulae and clusters, listing anything else in the frame too. Without ASTAP, or if the stars can't be matched, it falls back to the coordinates or target name the DWARF saved, and the card says so. The catalogue comes from OpenNGC (https://github.com/mattiaverga/OpenNGC), CC-BY-SA-4.0.

Every assessment also includes:
- **What's in the photo:** the main target, the constellation it's in, a short description, and other objects visible in the frame. For the Moon, it lists the named craters and maria you can see.
- **Light and colour (spectral overview):** how much of the object's light is red H-alpha, teal OIII, or plain starlight, and the mix of star colours from hot blue to cool red. The DWARF 3 records three broad colour channels, not a true spectrum, so this is worked out from colour ratios plus what's known about the object.
- **How it compares:** your best three earlier shots of the same target (by your rating, then by score) show up automatically. You can also add up to three reference images with **+ compare** in the file list. Claude then ranks them all and names the biggest gap.

  DWARFLAB's terms of service don't allow apps to pull photos from their site automatically. Instead, the comparison card links to the DWARFLAB photo wall, to r/dwarflabofficial and to AstroBin. Save any photo you want to measure up against, then add it with **+ compare**.

## Getting files straight from the telescope

**Over Wi-Fi:**
1. In the DWARFLAB app, put the DWARF 3 on your Wi-Fi (STA mode), or join the telescope's own hotspot with your laptop.
2. On the **From DWARF** tab, click **Find my DWARF**. The app checks every address on each local network your laptop is on (up to about 1,000 addresses per network), plus the devices your laptop has recently seen and the DWARF hotspot address 192.168.88.1. It doesn't just try the last-known address. It confirms the telescope by asking it for its device info, and saves its address. Nothing is sent outside your network.
3. Click **Load sessions** and pick a session. The stacked files are pre-selected. Click **Copy to laptop and measure**.

**Over USB-C:** plug the DWARF in and switch it on. It shows up as a drive, and the From DWARF tab switches to **USB-C** by itself. Load sessions and copy them the same way, with no Wi-Fi needed.

The DWARF 3 only offers file access over USB-C. Its firmware has no way to control the telescope through the cable, so goto, tracking, Stop all and Dwarfium always need Wi-Fi (home Wi-Fi or the telescope's hotspot).

Files go to `Pictures\DWARF Library` (you can change this in Settings). Nothing on the telescope is changed or deleted.

## Telescope control (Dwarfium inside the app)

The **Telescope** tab gives you full control of the DWARF from the app: goto and target search, tracking, the joystick, focus, calibration, live view, live stacking and schedules. It does this by running [Dwarfium](https://github.com/stevejcl/dwarfium), a free, open-source app (MIT license) that replicates the DWARFLAB app, inside the tab.

1. Open the **Telescope** tab and click **Install Dwarfium**.
   - The app downloads Dwarfium's browser edition (`Dwarfium-Win.zip`) from the official release page, which the Dwarfium README links to.
   - It checks the download against the SHA-256 checksum GitHub publishes for that file, and refuses to unpack it if they don't match.
   - Windows Defender scans it as usual.
2. Click **Start Dwarfium**. A console window opens for Dwarfium's helper services. Leave it open, but you can minimise it. If Windows asks about network access for *DwarfiumProxy* or *mediamtx*, allow it on private networks only.
3. Dwarfium appears in the tab. Click **Find my DWARF** (or **Fill in Dwarfium** if it's already found). The app fills in the DWARF's address, your site's latitude and longitude (set them in Settings, or click *Use this laptop's location* there) and your time zone, and opens Dwarfium's connect page. Press **Connect**.
4. If it says *Monitoring only*, the DWARFLAB phone app is holding control. Close the phone app, or turn off **Set Current Device as Host** in it, then click **Request control**.
5. For the first-time Bluetooth Wi-Fi setup, click **Open in its own window**, because Bluetooth doesn't work inside the tab.

You can also install the Dwarfium **desktop app** from the [Windows installer on its release page](https://github.com/acocalypso/dwarfium/releases). Once it's installed, an **Open Dwarfium** button appears at the top of this app. The installer isn't code-signed, so SmartScreen will warn you. Click **More info → Run anyway**.

## Plan & Control

A simpler alternative to Dwarfium for a night out.

- **Plan any night.** Pick a date. The site comes from your latitude and longitude in Settings. The app works out when it's properly dark and how bright the Moon is. It ranks about 85 DWARF-sized targets by how high they get and for how long, how far they are from the Moon, and how well they fit the DWARF's field of view. It also suggests an order for the night. Plans for future nights work the same way.
- **Go to** points the DWARF at a target in one click. It calibrates first if it needs to, then tracks. **Copy coordinates** copies the name, RA and Dec to paste into Dwarfium or the DWARFLAB app.
- **Camera settings.** Choose the filter, exposure, gain and frame count, or pick a recommended preset. Then click **Start capture**. When you're connected, the choices are read from your DWARF.
- **Moving the telescope.** Click the live view, then hold **W**/**S** to tilt up or down and **A**/**D** to turn left or right. Hold **Shift** for full speed. Letting go stops the telescope, and so does losing contact. Scroll over the live view to zoom, and double-click to reset.
- **Live view.** The DWARF 3 sends H.265 video. When Dwarfium's video relay is running (it starts with Dwarfium or the Telescope tab), live view plays through it, so the browser decodes the video with the graphics card, as Dwarfium does. Otherwise the app decodes the video itself, which uses more CPU and can look rougher.
- **Battery, temperature and storage.** The Telescope card shows the DWARF's battery level, its temperature (and the camera sensor's, when it reports it) and free space on its card. Battery and temperature show while the app is connected; storage shows whenever the DWARF is reachable.
- **Mount.** Choose Alt-Az for normal use or EQ for wedge mode. Alt-Az limits subs to 15 s, while EQ mode allows 30 to 60 s.

Goto needs your site's latitude and longitude in Settings.

## Network

**Network** opens as a panel over whatever you're doing (live view, a processing run), so nothing underneath stops; close it with **Close** or Esc. At the top, **Hardware usage** shows the processor (each thread), memory, the graphics card (NVIDIA), network traffic, free space on each drive (marking the library and processing drives), and the busiest programs, with the app and the tools it runs named.

The **Network** tab shows live, every 2 seconds:
- how the laptop reaches the DWARF: your Wi-Fi (STA mode), the DWARF's own hotspot, or not at all;
- whether it's online, the round-trip time, and whether the control port is open;
- whether the app's control link is up;
- a 2-minute round-trip graph, with missed checks marked in red;
- the laptop's own connections, wired and Wi-Fi, with Wi-Fi signal and link speed.

**Test all connections** measures every address the DWARF answers on: round trip, jitter, dropped connections, response time and download speed from its file server. The steadiest one is saved as the DWARF's address. **Find my DWARF** runs this test automatically after it finds the telescope.

If the DWARF isn't on your Wi-Fi, there are backup ways to reach it:
- **The DWARF's hotspot.** On Windows, the tab lists any DWARF hotspots in range, and **Join** connects to one. Leave the password blank to use the factory default. **Back to my Wi-Fi** returns the laptop to the network it was on before. A wired connection keeps working the whole time.
- **Bluetooth**, in Chrome or Edge. It shows which network the DWARF is on and its address, and can ask it to rejoin its saved Wi-Fi. These are the same setup messages the DWARFLAB app and Dwarfium use. Bluetooth can't carry control or pictures. Close the DWARFLAB phone app first, since only one device can connect at a time.
- **USB-C.** The tab shows when the DWARF's drive is plugged in. USB-C carries files only.

## Auto-apply a processing plan

After **Assess**, the Processing plan has an **Auto-apply plan** button. It shows how many raw frames the session has (for example "915 frames found") and where they are: the laptop library, the processing folder, or the DWARF itself over USB-C or Wi-Fi. Frames the DWARF rejected (`failed_*`) rank far below the ones it passed: with **Consider frames the DWARF rejected** ticked (it starts unticked), each one is measured (star count, sharpness, roundness) against a sample of the passed frames, and only those sharper and with more stars than the median passed frame get in, at most a tenth of the stack. The rest wait in `failed_frames` next to `lights`. Leave it unticked to keep them all out. Then it:

1. copies the missing frames into the **processing folder** (Settings), after checking there's room. Hundreds of frames need tens of GB, so pick a big drive, ideally an SSD. It also looks for the DWARF's own master dark for that exposure and gain (Astronomy/CALI_FRAME on the DWARF, over USB-C or Wi-Fi) and uses it only if its hot pixels line up with your frames;
2. stacks them in Siril: dark subtraction and cosmetic correction (when a dark fits), debayer, 2-pass registration against the **middle** frame of the session (so the sky's rotation spreads evenly both ways), keeps the sharpest frames that are also among the roundest 90%, rejection (Generalized ESD from 50 frames, which catches satellite trails; Winsorized below), frames weighted by sharpness (Siril 1.4), 32-bit;
3. keeps the **original frame size** (3840×2160): the DWARF's alt-az mount lets the sky turn in the frame over a long session, so corners only some frames covered are kept and any never covered are filled with the background level. Untick **Keep the original frame size** to crop to the area every frame shares instead;
4. GraXpert background extraction, on the graphics card when there is one (smoother for galaxies, so a big one like M 31 keeps its faint halo);
5. plate solves (Siril, offline with its local star catalogues in 1.4; ASTAP if Siril can't) and calibrates the colour with SPCC (the DWARF 3's IMX678 sensor with its UV/IR-cut Astro filter, or narrowband mode for the Duo-Band filter: Hα 656 nm/18 nm, OIII 501 nm/30 nm), else PCC, else from the image's own stars; then **neutralises the background** so it's grey, not green;
6. GraXpert's AI deconvolution, given the stack's measured star size: the object model (structure) then the stellar model (star cores, at most 0.3) for nebulae and galaxies, the other way round for clusters (other deconvolution did more harm than good on DWARF data). It needs GraXpert 3.1 or newer;
7. noise reduction: GraXpert's AI denoise at 0.7 under an hour of data, 0.6 up to three hours and 0.5 beyond (it softens stars) (Siril's NL-Bayes without GraXpert), then a colour-noise pass that smooths only the colour (it skips the stars, so their colour doesn't smear into coloured rings), stronger for Duo-Band, where only the red pixels (a quarter of the sensor) see Hα and come out much noisier;
8. the plan's own Siril commands, then a colour-keeping stretch: only the brightness is stretched, so the colour calibration survives, with the sky at 11% (12% for Duo-Band) and kept grey; then the faint nebula is lifted above the sky (the sky and stars untouched), and the faint sky is smoothed, brightness only, just until its grain is about 18% of the sky level (smoother looks plasticky); then saturation on the subject but not the sky or the stars, and the stars' colour is capped so they keep a tint instead of turning into red or cyan discs (also in HOO);
9. star reduction only with **StarNet++** (Setup links it); without it the stars are left alone. A **quality check** then measures the finished image: a background more than 5% off neutral is neutralised and uneven colour noise is smoothed, and the step note shows the numbers. Saves `result_final` as 16-bit TIFF, PNG, JPG and FITS. For Duo-Band sessions of emission nebulae the main result is the HOO palette (Hα red, OIII teal) and the natural-colour version is `result_natural`; reflection nebulae, galaxies and clusters keep natural colour as the main result. The plan's own stretch and saturation are left out in favour of the tuned ones.

**Reuse the last stack (redo only the finishing)** appears when a session already has a stack: it skips copying and stacking and redoes everything after it in a few minutes, handy after an update that improves the finishing.

Anything the plan doesn't set is chosen from the data: how many frames, the total integration time, the filter, and what kind of object it is. Each step's note says what it did and why, and if a program can't do a step, the next option is used instead of stopping the run.

A progress bar shows how far the whole run is and roughly how long is left. **PAUSE ALL** at the top suspends whatever program the app is running (Siril, GraXpert) and holds frame copying; press **RESUME ALL** to carry on exactly where it was.

**Tuned to each computer.** Every time the app starts it measures the computer it's on: processor threads and speed, memory, graphics card, and how fast the processing drive writes. From that it sets how many threads Siril uses (all but one), how much free memory it may take, whether GraXpert's AI runs on the graphics card (and its batch size), how many frames are copied at once, and whether processing runs at raised priority. The Network panel's Hardware usage section shows what it measured and chose.

Each step shows its progress. Nothing on the DWARF or in your library is changed. Afterwards, **Measure the result** assesses the new image, and you can delete Siril's working copies to free space.

### LAST SESSION

If a run doesn't finish (the app was closed or updated, the computer restarted, Stop all was pressed, or a step failed), press **LAST SESSION** at the top. It lists every unfinished run with where it stopped; **Resume** carries on from that step. Frames already copied aren't copied again, and a finished stack, crop or GraXpert result that's still on disk is kept, so a run cut off after a two-hour stack doesn't restack. Each run's progress is saved as `run.json` in its processing folder.

**Two drives (fast SSD for working, big hard drive for keeping).** Set **Processing folder** to a folder on a fast drive that isn't the system drive (for example `E:\IN-PROCESS` on a spare SSD) and **Finished runs folder** to a folder on a big hard drive (for example `D:\DWARF Processing`). Copying, stacking and every intermediate step then run on the SSD; when a run finishes, Siril's working copies are deleted and everything else (frames, stack, results, `run.json`) is moved to `<Finished runs folder>\<session>`, so the SSD is free for the next one. Re-running a session copies its frames back from there (only the stack when **reuse the last stack** is ticked), LAST SESSION looks in both folders, and if the SSD hasn't room for a run it works in the finished-runs folder instead. Stopped or failed runs stay on the SSD so Resume is quick. Leave Finished runs folder blank to keep everything in the processing folder. When the processing folder is on the system drive, the hardware check at startup suggests another drive.

## Diagnose

Click **Diagnose** at the top, say what's going wrong, and leave **Include a screenshot** ticked. Your browser asks which tab to share: pick the DWARF Post-Processing Hub tab. Claude then gets the screenshot plus the app's state:
- version and settings, without your API key;
- the DWARF connection and control link;
- Dwarfium's status;
- recent errors from the app and the page;
- recent background jobs;
- the last commands sent to the DWARF and its replies.

Every command the app sends to the DWARF, and each reply, is also written to `dwarf-commands.log` in the data folder. Wi-Fi details are never written there.

Claude replies with the likely cause and steps to fix it. It can also offer one-click fixes: find the DWARF, test connections, connect or hand back control, save an address, switch the mount, or open a tab or Settings. A fix only runs when you click it.

If the problem needs a change to the app itself, it says so. Each report is also saved to `%APPDATA%\DwarfOverview\diagnostics` (the last 10 are kept), so Claude can read the details when you ask in your project chat. Diagnose uses your API key, and needs the internet.

## Calibrate & learn

Recommendations start from DWARFLAB's DWARF 3 manual and community guides:
- 15–60 s subs, with EQ mode needed for 30 s and above.
- Gain 60–80, with 40 as the minimum.
- 200–400 frames.
- Duo-band for emission and planetary nebulae, and Astro for galaxies, clusters and dusty nebulae.

The guides are the DWARF 3 manual, the dwarfastro.com settings guide and New Forest Astro's filter guide.

With **Calibrate & learn** on, every capture started from the app is logged with its target and conditions. Every deep-sky photo you measure records how the night went: frames kept versus taken, signal-to-noise, star size and roundness, and your star rating. Once there are two or more results for a kind of target, the settings that worked best for you are recommended instead, marked "learned". The reasons are shown under the camera settings. If your sessions keep losing frames, a shorter exposure is suggested.

With **Auto-apply recommended settings** on, clicking **Go to** also fills in the camera settings for that target.

## Projects

The **Projects** tab keeps finished photos together. Each project is a real folder, by default in `Pictures\DWARF Projects`, and you can change that in Settings. Click **Add to project** on the Assess tab to copy a photo in. Each photo gets an overview built from the file and the DWARF's `shotsInfo.json`:
- the site and sky coordinates;
- the date;
- the file size and pixel dimensions;
- the frames stacked out of the frames taken;
- exposure and gain;
- the elapsed time from the first frame to the last.

Projects and photos both have notes, which save as you type.

## Themes

Settings has 30 earth tones. Pick a primary colour for highlights and tabs, and a secondary colour for the main buttons.

## SCAN & CONNECT, reboot and auto-reconnect

- **SCAN & CONNECT**, at the top, searches for the DWARF again and again until it finds it, then connects. It checks the last address first, then the hotspot, recently seen devices and the whole local network. If nothing is found within 30 seconds, it says so.
- **Reboot DWARF** is in Settings. It restarts the telescope. The DWARF is offline for 1–2 minutes, and the app waits for it, then reconnects by itself.
- **Auto-reconnect.** If the app's control link to the DWARF drops, the app searches for it 5 times, once a second:
  - first its last address;
  - then its hotspot, recently seen devices and the whole local network.

  If it's found, the link is restored, at a new address if needed. If not, a red banner says so, with a **Connect again** button. If the link keeps dropping while the DWARF is still on the network, the banner says another app is probably taking control. It doesn't fight that app for control.
- **Live view** now switches the DWARF's camera on by itself, the same way Dwarfium's Camera page does. You don't need to open Dwarfium first.

## Stop all

The red **Stop all** button at the top works from any tab. After you confirm, it sends the DWARF every stop command it has: joystick and motors, goto, calibration, EQ solving, tracking (including the Sun/Moon and sentry modes), live stacking on both cameras, bursts, videos, timelapses and focusing. It also cancels any GraXpert or Siril run in progress. A live stack keeps the frames it already has. The result lists what was stopped; "declined" usually means that thing wasn't running.

Stop all talks to the telescope directly over Wi-Fi, so it works even when Dwarfium isn't open. It needs the DWARF's address, which **Find my DWARF** sets.

## Updates

When a new version is out, a banner appears at the top of the app. You can also click the version number to check. **Install and restart** does the following:
1. Downloads the update from GitHub.
2. Checks its SHA-256 fingerprint.
3. Makes sure it only contains the app's own kinds of files.
4. Backs up your current version to `%APPDATA%\DwarfOverview\backups`.
5. Restarts the app.

Your settings, API key and history are never touched by an update.

## What it can and can't do yet

- GraXpert and Siril can't be opened straight to a particular dialog from outside the app. "Open" loads your image in the program, and the plan tells you the exact menu path and values. GraXpert's background extraction and denoise, and Siril's commands, run automatically with the suggested settings.
- StarNet star removal runs through Siril's `starnet` command, and only if you've set StarNet up inside Siril first.
- Colour calibration with SPCC/PCC needs plate solving in Siril. The FITS from the DWARF includes the coordinates, but Siril may still ask you for the target name the first time.

Night mode turns the whole app a dim orange to keep your eyes dark-adapted in the field.

## Where things are stored

`%APPDATA%\DwarfOverview` holds `config.json` (settings and your API key) and `history.db` (your session log). Back up `history.db` if you reinstall Windows, because it holds everything the app has learned.

If you'd rather not save the key in a file, set a Windows environment variable named `ANTHROPIC_API_KEY` instead. The app uses that first.

## Troubleshooting

- **"GraXpert wasn't found"**: open Settings and paste the full path to `GraXpert.exe`.
- **"siril-cli wasn't found"**: point Settings at `siril.exe`. `siril-cli.exe` must be in the same folder, which it is in a standard install.
- **"Reading MP4/AVI needs opencv"**: close the app and run `start-windows.bat` again. It installs anything that's missing.
- **Subject says "from the DWARF's target name"**: ASTAP isn't installed or couldn't match the stars. Install ASTAP and the D50 database, or set its path in Settings.
- **Port already in use**: something else is using port 8765. Close any other copy of the app first.
