import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pluvianus import npz_viewer_session as session


class NpzSessionFormatTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'settings.npzview.json'
        self.document = {
            'format': session.FORMAT, 'version': 1,
            'files': [{'path': 'data/spectrum.npz', 'mode': 'spectrogram',
                       'curves': [{'column': {'index': 0, 'label': 1.5}, 'axis': 'bottom_left',
                                   'visible': False, 'color': '#ff123456'}],
                       'spectrogram': {'axis': 'top_right', 'visible': True,
                                       'colormap': 'hot', 'levels': [1e-20, 1e-18]}}],
            'view': {'time_range': [12., 14.], 'time_auto_range': False,
                     'axes': {'bottom_left': {'range': [-2., 3.], 'auto_range': True}},
                     'splitters': {'sidebar': [.2, .8], 'plots': [.4, .6]}},
        }

    def test_json_round_trip_and_relative_references(self):
        self.document['files'][0]['path'] = 'data/árvíz.npz'
        session.write_session(self.path, self.document)
        self.assertEqual(session.read_session(self.path), self.document)
        self.assertIn('árvíz', self.path.read_text(encoding='utf-8'))
        source = self.path.parent / 'data' / 'árvíz.npz'
        reference = session.source_reference(source, self.path)
        self.assertEqual(reference, 'data/árvíz.npz')
        self.assertEqual(session.resolve_reference(reference, self.path), os.path.realpath(source))
        # Moving the settings and data directory together keeps references valid.
        moved = self.path.parent / 'moved' / self.path.name
        self.assertEqual(session.resolve_reference(reference, moved),
                         os.path.realpath(moved.parent / 'data' / 'árvíz.npz'))

    @unittest.skipUnless(os.name == 'nt', 'Different-drive paths are a Windows case.')
    def test_references_on_another_drive_are_absolute(self):
        reference = session.source_reference('Z:/recordings/lfp.npz', 'C:/sessions/settings.npzview.json')
        self.assertTrue(os.path.isabs(reference))
        self.assertEqual(reference, Path(os.path.realpath('Z:/recordings/lfp.npz')).as_posix())
        self.assertEqual(session.resolve_reference(reference, 'C:/sessions/settings.npzview.json'),
                         os.path.realpath('Z:/recordings/lfp.npz'))

    def test_failed_write_preserves_existing_settings_and_cleans_temporary_file(self):
        session.write_session(self.path, self.document)
        original = self.path.read_bytes()
        changed = copy.deepcopy(self.document)
        changed['files'] = []
        with patch('pluvianus.npz_viewer_session.os.replace', side_effect=OSError('write failed')):
            with self.assertRaises(OSError):
                session.write_session(self.path, changed)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_invalid_settings_are_rejected_before_writing(self):
        invalid = []
        for key, value in [('format', 'other'), ('version', 2), ('version', True), ('files', {})]:
            changed = copy.deepcopy(self.document)
            changed[key] = value
            invalid.append(changed)
        for key, value in [('mode', 'unknown'), ('path', ''), ('path', 'bad\0path')]:
            changed = copy.deepcopy(self.document)
            changed['files'][0][key] = value
            invalid.append(changed)
        for key, value in [('axis', 'left'), ('visible', 'yes'), ('levels', [3., 2.]),
                           ('levels', [0., float('nan')])]:
            changed = copy.deepcopy(self.document)
            changed['files'][0]['spectrogram'][key] = value
            invalid.append(changed)
        changed = copy.deepcopy(self.document)
        changed['files'][0]['curves'][0]['color'] = 'not-a-color'
        invalid.append(changed)
        changed = copy.deepcopy(self.document)
        changed['files'][0]['curves'][0]['column']['label'] = ['wrong', 'type']
        invalid.append(changed)
        changed = copy.deepcopy(self.document)
        changed['view']['splitters']['plots'] = [0., 0.]
        invalid.append(changed)
        for key in ('legends_visible', 'scale_bar_visible'):
            changed = copy.deepcopy(self.document)
            changed['view'][key] = 'yes'
            invalid.append(changed)
        for changed in invalid:
            with self.subTest(document=changed):
                with self.assertRaises(ValueError):
                    session.write_session(self.path, changed)
                self.assertFalse(self.path.exists())

    def test_nonfinite_json_and_invalid_json_are_rejected(self):
        for text in ['not json', '{"value": NaN}', '{"value": Infinity}']:
            self.path.write_text(text, encoding='utf-8')
            with self.subTest(text=text), self.assertRaises(ValueError):
                session.read_session(self.path)

    def test_optional_fields_and_unknown_additions_are_supported(self):
        document = {'format': session.FORMAT, 'version': 1, 'files': [], 'future_note': 'allowed'}
        session.write_session(self.path, document)
        self.assertEqual(json.loads(self.path.read_text(encoding='utf-8')), document)

    def test_interval_representation_is_optional_and_validated_in_version_one(self):
        curve = self.document['files'][0]['curves'][0]
        session.validate_session(self.document)
        curve['representation'] = 'intervals'
        session.write_session(self.path, self.document)
        self.assertEqual(session.read_session(self.path), self.document)
        self.assertEqual(self.document['version'], 1)
        curve['representation'] = 'invalid'
        with self.assertRaises(ValueError):
            session.validate_session(self.document)

    def test_spectrogram_db_scale_and_contrast_profiles_are_validated(self):
        image = self.document['files'][0]['spectrogram']
        image.update(scale='db', levels=[-110., -20.],
                     levels_by_scale={'linear': [1e-11, 1e-2], 'db': [-110., -20.]})
        session.write_session(self.path, self.document)
        self.assertEqual(session.read_session(self.path), self.document)
        image['scale'] = 'unknown'
        with self.assertRaises(ValueError):
            session.validate_session(self.document)
        image['scale'] = 'db'
        image['levels_by_scale']['db'] = [1., -1.]
        with self.assertRaises(ValueError):
            session.validate_session(self.document)


if __name__ == '__main__':
    unittest.main()
