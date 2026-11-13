"""Native Pynapple loading and column metadata shared by NPZ viewer modes."""

from dataclasses import dataclass

import numpy as np
import pynapple as nap


def native_scalar(value):
    if isinstance(value, np.generic) or (isinstance(value, np.ndarray) and value.ndim == 0):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='replace')
    return value


def metadata_text(value):
    value = native_scalar(value)
    return value.strip() or None if isinstance(value, str) else None


@dataclass(frozen=True)
class ColumnInfo:
    name: object
    data_unit: str | None = None
    data_label: str | None = None
    source_data_unit: str | None = None
    frequency_hz: float | None = None


def load_viewer_data(path):
    """Let Pynapple decode its timestamps, columns and native _metadata."""
    data = nap.load_file(str(path))
    if not isinstance(data, (nap.Tsd, nap.TsdFrame)):
        raise ValueError(f'Unsupported format: {type(data).__name__}')
    if isinstance(data, nap.TsdFrame) and data.shape[1] == 0:
        raise ValueError('The frame has no data columns.')
    if (not (np.issubdtype(data.values.dtype, np.number) or data.values.dtype == np.bool_)
            or np.iscomplexobj(data.values)):
        raise ValueError('Display data must be real numeric values.')
    return data


def column_info(data):
    if isinstance(data, nap.Tsd):
        return [ColumnInfo(name=None)]
    metadata = data.metadata
    result = []
    for index, name in enumerate(data.columns):
        row = metadata.iloc[index]
        frequency = native_scalar(row.get('frequency_hz'))
        try:
            frequency = float(frequency) if not isinstance(frequency, bool) else np.nan
        except (TypeError, ValueError):
            frequency = np.nan
        result.append(ColumnInfo(
            name=native_scalar(name), data_unit=metadata_text(row.get('data_unit')),
            data_label=metadata_text(row.get('data_label')),
            source_data_unit=metadata_text(row.get('source_data_unit')),
            frequency_hz=frequency if np.isfinite(frequency) and frequency >= 0 else None,
        ))
    return result


def frequency_coordinates(data):
    """Only explicit native frequency_hz metadata identifies a frequency axis."""
    if not isinstance(data, nap.TsdFrame) or data.shape[1] < 2:
        return None
    frequencies = [info.frequency_hz for info in column_info(data)]
    if any(value is None for value in frequencies):
        return None
    return np.asarray(frequencies, dtype=float)


def common_metadata(infos, key):
    values = {getattr(info, key) for info in infos if getattr(info, key) is not None}
    return next(iter(values)) if len(values) == 1 else None
