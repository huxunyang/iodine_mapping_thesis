import zipfile
import numpy as np

def _split_nrrd(data: bytes):
    sep = b'\\n\\n'
    idx = data.find(sep)
    if idx < 0:
        raise ValueError('NRRD header separator not found')
    header = data[:idx].decode('ascii', errors='replace')
    blob = data[idx + 2:]
    return header, blob

def _parse_header(header: str):
    fields = {}
    for line in header.splitlines():
        if line.startswith('NRRD') or line.startswith('#') or not line.strip():
            continue
        if ':' in line:
            k, v = line.split(':', 1)
            fields[k.strip()] = v.strip()
    return fields

def read_nrrd_from_zip(zip_path, member):
    with zipfile.ZipFile(zip_path, 'r') as z:
        data = z.read(member)

    header, blob = _split_nrrd(data)
    fields = _parse_header(header)

    sizes = list(map(int, fields['sizes'].split()))
    n = int(np.prod(sizes))

    dtype_map = {
        'float': np.float32,
        'double': np.float64,
        'short': np.int16,
        'int': np.int32,
        'uchar': np.uint8,
        'ushort': np.uint16,
    }
    dt = dtype_map.get(fields.get('type', 'float'), np.float32)

    enc = fields.get('encoding', 'raw').lower()
    if enc != 'raw':
        raise ValueError(f'Only raw encoding supported, got: {enc}')

    endian = fields.get('endian', 'little').lower()
    arr = np.frombuffer(blob, dtype=dt, count=n)
    if endian == 'big':
        arr = arr.byteswap().newbyteorder()

    arr = arr.reshape(sizes, order='C')
    return arr, fields

def list_members(zip_path, prefix=''):
    with zipfile.ZipFile(zip_path, 'r') as z:
        return [n for n in z.namelist() if n.startswith(prefix)]
