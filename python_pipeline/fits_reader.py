#!/usr/bin/env python3
"""Minimal FITS reader for SoLEXS data (no astropy dependency).

FITS format:
- 2880-byte header blocks (ASCII key=value)
- 2880-byte data blocks (binary)
- Header ends with "END" keyword
- Each header card is 80 bytes

For SoLEXS .lc files:
- Primary HDU: header only (no data)
- Extension 1 (RATE): BinTable with columns TIME (D=float64) and COUNTS (D=float64)
- 86400 rows

For SoLEXS .pi files:
- Primary HDU: header only
- Extension 1 (SPECTRUM): BinTable with TSTART, TELAPSE, SPEC_NUM, CHANNEL (340I),
  COUNTS (340E), EXPOSURE
- 86400 rows
"""
import numpy as np
import struct


def read_fits_header(f, max_blocks=100):
    """Read FITS header from open file. Returns (header_dict, n_axis, axis_sizes).
    
    A FITS header consists of one or more 2880-byte blocks of 80-byte ASCII
    cards. The header ends when an "END" card is found. We must read ALL
    blocks up to and including the one with END, then stop.
    """
    header = {}
    axis_sizes = []
    naxis = 0
    for _ in range(max_blocks):
        block = f.read(2880)
        if len(block) < 2880:
            break
        end_found = False
        for i in range(0, 2880, 80):
            card = block[i:i+80].decode("ascii", errors="replace").rstrip()
            if card.startswith("END") and "EXTEND" not in card and "END" in card.split()[0:1]:
                end_found = True
                break
            if "=" in card:
                key, val = card.split("=", 1)
                key = key.strip()
                val = val.strip()
                # Strip comment
                if "/" in val:
                    val = val.split("/")[0].strip()
                # Strip quotes
                if val.startswith("'") and val.endswith("'"):
                    val = val[1:-1].strip()
                else:
                    try:
                        if "." in val or "E" in val or "e" in val:
                            val = float(val)
                        else:
                            val = int(val)
                    except (ValueError, TypeError):
                        pass
                header[key] = val
                if key == "NAXIS":
                    naxis = int(val)
                elif key.startswith("NAXIS") and key[5:].isdigit():
                    axis_sizes.append(int(val))
        if end_found:
            break
    return header, naxis, axis_sizes


def read_fits_table(f, header):
    """Read a FITS binary table extension. Returns dict of column_name -> array."""
    naxis1 = header.get("NAXIS1", 0)  # bytes per row
    naxis2 = header.get("NAXIS2", 0)  # number of rows
    n_fields = header.get("TFIELDS", 0)
    
    if naxis1 == 0 or naxis2 == 0 or n_fields == 0:
        return {}
    
    # Parse column descriptions
    columns = []
    for i in range(1, n_fields + 1):
        name = header.get(f"TTYPE{i}", f"COL{i}")
        tform = header.get(f"TFORM{i}", "")
        # Parse TFORM: e.g., "D" = 1 float64, "8E" = 8 float32, "340E" = 340 float32
        # Format: [repeat]type
        repeat = 1
        type_char = tform[-1] if tform else ""
        if tform[:-1].isdigit():
            repeat = int(tform[:-1])
        
        type_map = {
            "L": ("?", 1),    # logical (bool)
            "B": ("u1", 1),   # unsigned byte
            "I": ("i2", 2),   # int16 (big-endian in FITS)
            "J": ("i4", 4),   # int32 (big-endian in FITS)
            "K": ("i8", 8),   # int64 (big-endian in FITS)
            "E": ("f4", 4),   # float32 (big-endian in FITS)
            "D": ("f8", 8),   # float64 (big-endian in FITS)
            "A": ("S1", 1),   # char
        }
        if type_char in type_map:
            dtype, item_size = type_map[type_char]
            # FITS uses big-endian byte order
            dtype = ">" + dtype
            byte_size = repeat * item_size
            columns.append((name, dtype, repeat, byte_size))
    
    # Read data
    data_size = naxis1 * naxis2
    # Round up to multiple of 2880
    data_blocks = (data_size + 2879) // 2880
    raw = f.read(data_blocks * 2880)
    if len(raw) < data_size:
        return {}
    
    # Build a numpy structured dtype matching the FITS binary table layout
    # Each column is at a specific byte offset within the row
    np_fields = []
    offsets = []
    offset = 0
    for name, dtype, repeat, byte_size in columns:
        if repeat == 1:
            np_fields.append((name.strip(), dtype))
        else:
            np_fields.append((name.strip(), dtype, (repeat,)))
        offsets.append(offset)
        offset += byte_size
    
    # Use numpy structured array for fast parsing
    # Construct dtype with explicit itemsize = NAXIS1 (in case there's padding)
    dt = np.dtype(np_fields)
    if dt.itemsize < naxis1:
        # Add padding to match row size
        np_fields.append(("_padding", f"V{naxis1 - dt.itemsize}"))
        dt = np.dtype(np_fields)
    
    # Read entire table as a structured array
    table = np.frombuffer(raw[:data_size], dtype=dt, count=naxis2)
    
    # Extract each column into a regular array (convert to native byte order)
    result = {}
    for name, dtype, repeat, byte_size in columns:
        col_name = name.strip()
        col_data = table[col_name]
        # Convert from big-endian to native byte order
        if repeat == 1:
            result[col_name] = np.array(col_data, dtype=dtype.replace(">", "")).astype(dtype.replace(">", ""))
        else:
            result[col_name] = np.array(col_data, dtype=dtype.replace(">", "")).astype(dtype.replace(">", ""))
    
    return result


def read_fits_file(path):
    """Read a FITS file and return list of (header, data) HDUs."""
    hdus = []
    with open(path, "rb") as f:
        while True:
            try:
                header, naxis, axis_sizes = read_fits_header(f)
            except Exception:
                break
            if not header:
                break
            
            # Determine data size
            if naxis > 0 and axis_sizes:
                # Compute total data size
                bitpix = header.get("BITPIX", 8)
                bytes_per_val = abs(bitpix) // 8
                n_vals = 1
                for s in axis_sizes:
                    n_vals *= s
                data_size = n_vals * bytes_per_val
            else:
                data_size = 0
            
            # Check if this is a binary table extension
            xtension = header.get("XTENSION", "")
            if xtension == "BINTABLE":
                data = read_fits_table(f, header)
            elif data_size > 0:
                # Read raw data array
                data_blocks = (data_size + 2879) // 2880
                raw = f.read(data_blocks * 2880)
                bitpix = header.get("BITPIX", 8)
                dtype_map = {8: "u1", 16: "i2", 32: "i4", 64: "i8",
                             -32: "f4", -64: "f8"}
                dtype = dtype_map.get(bitpix, "u1")
                data = np.frombuffer(raw[:data_size], dtype=dtype)
                if naxis > 1:
                    data = data.reshape(axis_sizes)
            else:
                data = None
            
            hdus.append((header, data))
            
            # Check for END card
            if "EXTEND" not in header and "XTENSION" not in header:
                # Primary HDU only
                pass
            
            # Peek at next block to see if there's another HDU
            pos = f.tell()
            peek = f.read(8)
            f.seek(pos)
            if not peek or peek == b"\x00" * 8:
                break
    
    return hdus


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/z/my-project/data/real_aditya_l1/20260615/AL1_SLX_L1_20260615_v1.0/SDD2/AL1_SOLEXS_20260615_SDD2_L1.lc"
    
    print(f"Reading: {path}")
    hdus = read_fits_file(path)
    print(f"HDUs: {len(hdus)}")
    for i, (hdr, data) in enumerate(hdus):
        print(f"\n--- HDU {i} ---")
        print(f"  XTENSION: {hdr.get('XTENSION', 'PRIMARY')}")
        print(f"  TELESCOP: {hdr.get('TELESCOP', '?')}")
        print(f"  INSTRUME: {hdr.get('INSTRUME', '?')}")
        print(f"  ORIGIN: {hdr.get('ORIGIN', '?')}")
        print(f"  DATE-OBS: {hdr.get('DATE-OBS', '?')}")
        if isinstance(data, dict):
            print(f"  Columns: {list(data.keys())}")
            for k, v in data.items():
                if hasattr(v, 'shape'):
                    print(f"    {k}: shape={v.shape}, dtype={v.dtype}, "
                          f"min={v.min()}, max={v.max()}")
                else:
                    print(f"    {k}: {v}")
        elif data is not None:
            print(f"  Data shape: {data.shape}")
