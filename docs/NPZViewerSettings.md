# NPZ viewer settings format

The NPZ viewer saves sessions as UTF-8 JSON files with the `.npzview.json`
extension. They reference the recordings; they do not embed or modify NPZ data.

Use **File → Save settings…** (`Ctrl+S`) to choose a file the first time and
update it subsequently. **Save settings as…** (`Ctrl+Shift+S`) chooses another
destination. **Load settings…** (`Ctrl+Shift+O`) replaces the current display with
the saved session.

Saved and loaded settings files also appear in **File → Open recent** alongside
NPZ recordings. Choosing a JSON entry restores the session; choosing an NPZ entry
opens that recording. The most recent 20 entries are remembered.

A minimal example:

```json
{
  "format": "pluvianus.npzview",
  "version": 1,
  "files": [
    {
      "path": "data/lfp_spectrum.npz",
      "mode": "spectrogram",
      "spectrogram": {
        "axis": "top_left",
        "visible": true,
        "colormap": "hot",
        "levels": [0.12, 8.4]
      }
    }
  ],
  "view": {
    "time_range": [120.0, 180.0],
    "time_auto_range": false,
    "legends_visible": true,
    "scale_bar_visible": true,
    "axes": {
      "top_left": {"range": [1.0, 200.0], "auto_range": false}
    },
    "splitters": {"sidebar": [0.25, 0.75], "plots": [0.5, 0.5]}
  }
}
```

`format`, `version`, and `files` are required. Unsupported versions and malformed
settings are rejected before changing the display. Other fields are optional;
unknown additional fields are ignored.

Files appear in list order. Paths use forward slashes and are relative to the
settings file where possible, allowing a settings/data folder to be moved
together. References on a different Windows drive remain absolute. Saving under
a new filename recalculates the references for its directory.

Each file has a `mode` of `curves` (default) or `spectrogram`. Its optional
`curves` list contains entries such as:

```json
{
  "column": {"index": 0, "label": "neuron_01"},
  "axis": "bottom_left",
  "representation": "curve",
  "visible": true,
  "color": "#ff1f77b4"
}
```

Column labels retain their JSON string, number, or boolean types. A standalone
`Tsd` uses column index `0` and label `null`. Colors are hex RGB (`#RRGGBB`) or Qt
ARGB (`#AARRGGBB`); the writer uses ARGB to preserve transparency. Reordered
columns are matched by label and reported; missing or ambiguous matches are
skipped rather than assigning their settings to another curve.

The optional per-curve `representation` is `curve` (the default for older files)
or `intervals`. This extension uses the existing schema version **1**. In interval
mode, the stored `axis` remembers the curve's regular destination; the shading
appears in both panels and uses a faint version of the stored color. Visibility
controls both panels together. Epochs are reconstructed from the NPZ data rather
than copied into the settings file. If a changed source no longer has two levels,
loading reports the change and displays the ordinary curve.

For two-level signals, the greater finite level is high. Constant zero/one
signals are supported as all-low/all-high. Each state holds from its timestamp
until the next sample, or the end of its recording interval. Missing samples and
`time_support` gaps interrupt the shading. Shading spans the full panel height,
follows zooming, and does not contribute to autoscaling.

Axis identifiers are `top_left`, `top_right`, `bottom_left`, and `bottom_right`.
Spectrogram settings apply to every image piece together. The writer records
the actual color limits, preserving the saved appearance; resetting colors in
the viewer recalculates the default percentiles. Both representations' settings
are saved, including an inactive spectrogram's previously chosen colors and the
individual frequency curves. Hidden image pieces are reconstructed lazily.

The view records the shared time range, its autoscale flag, all four vertical
ranges and autoscale flags, splitter proportions, and the global legend/time
scale-bar visibility. The optional boolean fields `legends_visible` and
`scale_bar_visible` use the existing schema version **1**; when omitted, the
current View-menu choices are retained. Omitted ranges autoscale.

When sources are missing, a dialog allows individual files to be located or the
available files to be loaded. Cancel leaves the current session intact. Loading
reports missing/unreadable files and changed columns together. If no referenced
recordings can load, the current session is retained. An empty `files` list is
valid and clears the display. An incompatible spectrogram grid falls back to
curves, and an unavailable colormap falls back to hot, with an explanation.

Saving validates the document and writes a temporary file beside the destination
before replacing it atomically. Saving captures the files currently open in the
viewer. Recent-file history and window placement remain in the application's
local preferences.
