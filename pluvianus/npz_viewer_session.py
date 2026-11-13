"""Versioned JSON session files for the NPZ viewer, independent of Qt."""

import json
import math
import os
from pathlib import Path
import re
import tempfile


FORMAT = 'pluvianus.npzview'
VERSION = 1
AXES = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
SUFFIX = '.npzview.json'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _range(value, name):
    _require(isinstance(value, list) and len(value) == 2
             and all(_number(v) for v in value) and value[0] < value[1],
             f'{name} must contain two finite numbers, minimum < maximum.')


def _display(entry, name):
    _require(isinstance(entry, dict), f'{name} must be an object.')
    _require(entry.get('axis', 'top_left') in AXES, f'{name}: unknown axis.')
    _require(type(entry.get('visible', True)) is bool, f'{name}: visibility must be true or false.')


def validate_session(document):
    """Validate known version-1 fields; leave future optional fields alone."""
    _require(isinstance(document, dict), 'Settings must be a JSON object.')
    _require(document.get('format') == FORMAT, 'This is not an NPZ viewer settings file.')
    _require(type(document.get('version')) is int and document['version'] == VERSION,
             f'Unsupported settings version: {document.get("version")!r}.')
    files = document.get('files')
    _require(isinstance(files, list), 'Settings must contain a files list.')
    for index, entry in enumerate(files):
        name = f'File {index + 1}'
        _require(isinstance(entry, dict), f'{name} must be an object.')
        _require(isinstance(entry.get('path'), str) and bool(entry['path'].strip())
                 and '\0' not in entry['path'], f'{name}: invalid path.')
        _require(entry.get('mode', 'curves') in ('curves', 'spectrogram'), f'{name}: unknown display mode.')
        curves = entry.get('curves', [])
        _require(isinstance(curves, list), f'{name}: curves must be a list.')
        indices = set()
        for curve in curves:
            _display(curve, name + ' curve')
            _require(curve.get('representation', 'curve') in ('curve', 'intervals'),
                     f'{name}: unknown curve representation.')
            column = curve.get('column')
            _require(isinstance(column, dict), f'{name}: missing curve column.')
            position = column.get('index')
            _require(type(position) is int and position >= 0 and position not in indices,
                     f'{name}: curve column indices must be distinct nonnegative integers.')
            indices.add(position)
            _require('label' in column, f'{name}: missing column label.')
            label = column['label']
            _require(label is None or type(label) in (str, bool) or _number(label),
                     f'{name}: column labels must be strings, finite numbers, booleans, or null.')
            color = curve.get('color', '#1f77b4')
            _require(isinstance(color, str) and re.fullmatch(r'#[0-9a-fA-F]{6}(?:[0-9a-fA-F]{2})?', color),
                     f'{name}: curve colors must be hex RGB or ARGB strings.')
        if 'spectrogram' in entry:
            image = entry['spectrogram']
            _display(image, name + ' spectrogram')
            _require(isinstance(image.get('colormap', 'hot'), str)
                     and bool(image.get('colormap', 'hot')), f'{name}: invalid colormap.')
            if image.get('levels') is not None:
                _range(image['levels'], name + ' color limits')
            _require(image.get('scale', 'linear') in ('linear', 'db'), f'{name}: unknown spectrogram scale.')
            scale_levels = image.get('levels_by_scale', {})
            _require(isinstance(scale_levels, dict), f'{name}: levels_by_scale must be an object.')
            for scale, limits in scale_levels.items():
                _require(scale in ('linear', 'db'), f'{name}: unknown contrast scale.')
                _range(limits, name + ' ' + scale + ' color limits')
    view = document.get('view', {})
    _require(isinstance(view, dict), 'View settings must be an object.')
    for key in ('legends_visible', 'scale_bar_visible'):
        _require(type(view.get(key, True)) is bool, f'{key} must be true or false.')
    if 'time_range' in view:
        _range(view['time_range'], 'Time range')
    _require(type(view.get('time_auto_range', False)) is bool, 'Time autoscale must be true or false.')
    axes = view.get('axes', {})
    _require(isinstance(axes, dict), 'View axes must be an object.')
    for name, axis in axes.items():
        _require(name in AXES and isinstance(axis, dict), f'Unknown or invalid axis: {name}.')
        if 'range' in axis:
            _range(axis['range'], name + ' range')
        _require(type(axis.get('auto_range', False)) is bool, f'{name}: autoscale must be true or false.')
    splitters = view.get('splitters', {})
    _require(isinstance(splitters, dict), 'Splitter settings must be an object.')
    for name in ('sidebar', 'plots'):
        if name in splitters:
            proportions = splitters[name]
            _require(isinstance(proportions, list) and len(proportions) == 2
                     and all(_number(v) and v >= 0 for v in proportions)
                     and _number(sum(proportions)) and sum(proportions) > 0,
                     f'{name}: splitter proportions must be two nonnegative numbers with a positive sum.')
    return document


def source_reference(source, session_path):
    """Use a portable relative reference, or an absolute path on another drive."""
    # Windows temporary/profile folders may be reached through short names or
    # junctions. Compare canonical paths on both sides before making a reference.
    source = os.path.realpath(source)
    try:
        reference = os.path.relpath(source, os.path.dirname(os.path.realpath(session_path)))
    except ValueError:  # Windows paths on different drives.
        reference = source
    return Path(reference).as_posix()


def resolve_reference(reference, session_path):
    return os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(session_path)), reference))


def _invalid_constant(value):
    raise ValueError(f'Non-finite JSON number: {value}.')


def read_session(path):
    with open(path, encoding='utf-8') as stream:
        return validate_session(json.load(stream, parse_constant=_invalid_constant))


def write_session(path, document):
    """Replace the destination atomically after validation and a complete write."""
    validate_session(document)
    contents = json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + '\n'
    path = Path(path).absolute()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                         dir=path.parent, prefix='.' + path.name + '.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
