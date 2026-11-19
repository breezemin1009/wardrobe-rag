"""Read selected members of a publicly shared author's Google Drive archive.

Uses the normal large-file confirmation form and standard byte-range requests.
Login pages, denied access, unsupported ranges and unexpected hosts are errors.
No downloaded program or pickle is executed by this module.
"""
import io
import struct
import zlib
from html.parser import HTMLParser
from urllib.parse import urlparse
import requests


class Confirmation(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action, self.fields = None, {}

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == 'form':
            self.action = values.get('action')
        if tag == 'input' and values.get('name'):
            self.fields[values['name']] = values.get('value', '')


def public_download(identifier):
    session = requests.Session()
    response = session.get('https://drive.google.com/uc', params={'export': 'download', 'id': identifier}, timeout=30)
    response.raise_for_status()
    parser = Confirmation()
    parser.feed(response.text)
    if 'Virus scan warning' not in response.text or parser.action != 'https://drive.usercontent.google.com/download':
        raise ValueError('The author archive is not available through its public confirmation form')
    if parser.fields.get('id') != identifier:
        raise ValueError('Archive identifier changed')
    with session.get(parser.action, params=parser.fields, headers={'Range': 'bytes=0-0'}, stream=True, timeout=30) as result:
        result.raise_for_status()
        if result.headers.get('Content-Type', '').startswith('text/'):
            raise ValueError('Public archive is unavailable or requires sign-in')
        length = int(result.headers['Content-Range'].rsplit('/', 1)[1]) if result.status_code == 206 else int(result.headers['Content-Length'])
        url = result.url
        if urlparse(url).scheme != 'https' or urlparse(url).hostname != 'drive.usercontent.google.com':
            raise ValueError('Unexpected public archive host')
    return session, url, length


class RangeArchive(io.RawIOBase):
    def __init__(self, session, url, size):
        self.session, self.url, self.size, self.position = session, url, size, 0
        self.bytes_received = 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if position < 0:
            raise ValueError('Negative archive offset')
        self.position = position
        return position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0: return b''
        if size > 256 * 1024 * 1024:
            raise ValueError('Use bounded reads for large archive members')
        start, end = self.position, self.position + size - 1
        with self.session.get(self.url, headers={'Range': f'bytes={start}-{end}'}, stream=True, timeout=45) as response:
            if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                raise ValueError('This public server does not support verified range reads')
            result = response.content
            if len(result) != size:
                raise ValueError('Incomplete archive range')
        self.position += size
        self.bytes_received += size
        return result


def read_indexed_member(archive, entry, maximum=16 * 1024 * 1024):
    """Read a bounded indexed JSON/image entry and verify ZIP's length and CRC."""
    if entry['flags'] & 1 or entry['method'] not in [0, 8]:
        raise ValueError('Unsupported archive entry')
    if not 0 <= entry['bytes'] <= maximum or not 0 <= entry['compressed'] <= maximum:
        raise ValueError('Archive entry exceeds the explicit read limit')
    archive.seek(entry['offset'])
    prefix = archive.read(min(4096, archive.size - entry['offset']))
    fields = struct.unpack_from('<4s5H3I2H', prefix)
    if fields[0] != b'PK\x03\x04' or fields[3] != entry['method']:
        raise ValueError('Archive local header differs from its index')
    name_size, extra_size = fields[-2:]
    local_name = prefix[30:30 + name_size].decode('utf8' if fields[2] & 2048 else 'cp437')
    if local_name != entry['path']:
        raise ValueError('Archive path differs from its index')
    start = 30 + name_size + extra_size
    data = prefix[start:start + entry['compressed']]
    if len(data) < entry['compressed']:
        archive.seek(entry['offset'] + start + len(data))
        data += archive.read(entry['compressed'] - len(data))
    if entry['method'] == 8:
        decoder = zlib.decompressobj(-15)
        result = decoder.decompress(data, entry['bytes'] + 1)
        if not decoder.eof or decoder.unused_data:
            raise ValueError('Invalid compressed archive entry')
    else:
        result = data
    if len(result) != entry['bytes'] or zlib.crc32(result) != entry['crc32']:
        raise ValueError('Archive member integrity check failed')
    return result
