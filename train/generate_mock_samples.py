"""
Standalone Pure-Python PNG Generator (Zero External Dependencies).
Creates valid PNG sample images for out-of-the-box testing without requiring Pillow.
"""

import zlib
import struct
import os

def create_pure_png(width: int, height: int, r: int, g: int, b: int) -> bytes:
    """Generates an uncompressed valid PNG image of given dimensions and solid color."""
    # 8-byte signature
    signature = b"\x89PNG\r\n\x1a\n"
    
    # IHDR chunk: width (4 bytes), height (4 bytes), bit depth (1 byte), color type (1 byte: 2=RGB), compression (1), filter (1), interlace (1)
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data)
    ihdr_chunk = struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
    
    # Scanlines: each row starts with filter byte 0 followed by width * 3 bytes (RGB)
    row = b"\x00" + bytes([r, g, b]) * width
    raw_data = row * height
    compressed_data = zlib.compress(raw_data, level=6)
    
    # IDAT chunk
    idat_crc = zlib.crc32(b"IDAT" + compressed_data)
    idat_chunk = struct.pack(">I", len(compressed_data)) + b"IDAT" + compressed_data + struct.pack(">I", idat_crc)
    
    # IEND chunk
    iend_crc = zlib.crc32(b"IEND")
    iend_chunk = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
    
    return signature + ihdr_chunk + idat_chunk + iend_chunk

def generate_samples(target_dir: str):
    os.makedirs(target_dir, exist_ok=True)
    
    # Sample Invoice (White background with slight cool tint)
    inv_data = create_pure_png(800, 1000, 250, 250, 252)
    with open(os.path.join(target_dir, "sample_invoice.png"), "wb") as f:
        f.write(inv_data)
        
    # Sample Receipt (Warm paper tint)
    rec_data = create_pure_png(450, 700, 254, 253, 248)
    with open(os.path.join(target_dir, "sample_receipt.png"), "wb") as f:
        f.write(rec_data)
        
    # Sample Medical Form (Clean white form)
    form_data = create_pure_png(800, 950, 255, 255, 255)
    with open(os.path.join(target_dir, "sample_medical_form.png"), "wb") as f:
        f.write(form_data)
        
    print(f"Generated sample PNG documents in: {target_dir}")

if __name__ == "__main__":
    generate_samples("samples")
