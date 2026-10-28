"""
Diverse High-Fidelity Receipt Generator.
Renders 4 distinct, realistic scanned/printed style receipts:
1. Coffee Shop / Cafe Receipt (Starbeans Coffee)
2. Fine Dining Restaurant Bill (The Golden Fork Bistro)
3. Supermarket Grocery Receipt (FreshMart Superstore)
4. Electronics & Tech Retail Bill (TechZone Electronics)
"""

import os
from PIL import Image, ImageDraw, ImageFont

def get_font(size: int = 14):
    try:
        # Try standard TrueType fonts available on macOS/Linux
        for font_path in [
            "/System/Library/Fonts/Supplemental/Courier New.ttf",
            "/System/Library/Fonts/Courier.dfont",
            "/System/Library/Fonts/Menlo.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/freefont/FreeMono.ttf"
        ]:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, size)
    except Exception:
        pass
    return ImageFont.load_default()

def draw_receipt_cafe(target_path: str):
    w, h = 480, 780
    img = Image.new("RGB", (w, h), color=(253, 252, 248))
    draw = ImageDraw.Draw(img)
    font_lg = get_font(18)
    font_md = get_font(13)
    font_sm = get_font(11)

    # Header
    draw.text((w // 2 - 130, 30), "STARBEANS COFFEE & BAKERY", fill=(30, 30, 30), font=font_lg)
    draw.text((w // 2 - 85, 58), "Store #8802 - Seattle, WA", fill=(80, 80, 80), font=font_sm)
    draw.text((w // 2 - 65, 75), "Tel: (206) 555-0199", fill=(100, 100, 100), font=font_sm)

    # Dashed divider
    draw.line([(30, 100), (w - 30, 100)], fill=(180, 180, 180), width=1)

    # Metadata
    draw.text((35, 115), "ORDER #: ORD-88219", fill=(30, 30, 30), font=font_md)
    draw.text((35, 138), "DATE:    2026-10-28 09:14 AM", fill=(40, 40, 40), font=font_sm)
    draw.text((35, 158), "CASHIER: Alex M.", fill=(40, 40, 40), font=font_sm)
    draw.text((35, 178), "TYPE:    Dine-In", fill=(40, 40, 40), font=font_sm)

    draw.line([(30, 205), (w - 30, 205)], fill=(180, 180, 180), width=1)

    # Items Header
    draw.text((35, 215), "QTY  ITEM DESCRIPTION               PRICE", fill=(60, 60, 60), font=font_sm)
    draw.line([(30, 235), (w - 30, 235)], fill=(180, 180, 180), width=1)

    items = [
        (" 1   Caramel Macchiato Grande", "$5.45"),
        (" 1   Iced Vanilla Latte 16oz", "$4.95"),
        (" 2   Artisan Butter Croissant", "$6.50"),
        (" 1   Organic Blueberry Muffin", "$3.25"),
    ]

    y = 250
    for desc, prc in items:
        draw.text((35, y), desc, fill=(20, 20, 20), font=font_sm)
        draw.text((w - 95, y), prc, fill=(20, 20, 20), font=font_sm)
        y += 28

    draw.line([(30, y + 10), (w - 30, y + 10)], fill=(180, 180, 180), width=1)
    y += 25

    # Subtotals
    draw.text((w - 230, y), "SUBTOTAL:", fill=(50, 50, 50), font=font_sm)
    draw.text((w - 95, y), "$20.15", fill=(30, 30, 30), font=font_sm)
    y += 22

    draw.text((w - 230, y), "TAX (8.5%):", fill=(50, 50, 50), font=font_sm)
    draw.text((w - 95, y), "$1.71", fill=(30, 30, 30), font=font_sm)
    y += 28

    # Total Double Line
    draw.line([(w - 240, y), (w - 30, y)], fill=(30, 30, 30), width=2)
    y += 8
    draw.text((w - 230, y), "TOTAL:", fill=(20, 20, 20), font=font_lg)
    draw.text((w - 110, y), "$21.86", fill=(200, 30, 30), font=font_lg)
    y += 28
    draw.line([(w - 240, y), (w - 30, y)], fill=(30, 30, 30), width=2)

    # Payment footer
    y += 35
    draw.text((w // 2 - 95, y), "PAID BY: VISA ENDING *4092", fill=(60, 60, 60), font=font_sm)
    draw.text((w // 2 - 80, y + 20), "AUTH CODE: 091428", fill=(80, 80, 80), font=font_sm)
    draw.text((w // 2 - 105, y + 55), "Thank you for visiting Starbeans!", fill=(80, 80, 80), font=font_sm)

    img.save(target_path)
    return target_path

def draw_receipt_restaurant(target_path: str):
    w, h = 500, 850
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font_lg = get_font(18)
    font_md = get_font(13)
    font_sm = get_font(11)

    draw.text((w // 2 - 120, 35), "THE GOLDEN FORK BISTRO", fill=(20, 20, 20), font=font_lg)
    draw.text((w // 2 - 100, 65), "442 Park Avenue, Suite 10", fill=(70, 70, 70), font=font_sm)
    draw.text((w // 2 - 80, 85), "New York, NY 10022", fill=(70, 70, 70), font=font_sm)
    draw.text((w // 2 - 70, 105), "GST/VAT: US8819201", fill=(90, 90, 90), font=font_sm)

    draw.line([(35, 130), (w - 35, 130)], fill=(150, 150, 150), width=1)

    draw.text((40, 145), "CHECK #: CHK-49102", fill=(20, 20, 20), font=font_md)
    draw.text((40, 170), "TABLE:   Table 12 (Guests: 2)", fill=(40, 40, 40), font=font_sm)
    draw.text((40, 190), "SERVER:  Marcus G.", fill=(40, 40, 40), font=font_sm)
    draw.text((40, 210), "DATE:    2026-10-28 08:30 PM", fill=(40, 40, 40), font=font_sm)

    draw.line([(35, 235), (w - 35, 235)], fill=(150, 150, 150), width=2)
    draw.text((40, 245), "ITEM DESCRIPTION            QTY    AMOUNT", fill=(50, 50, 50), font=font_sm)
    draw.line([(35, 265), (w - 35, 265)], fill=(150, 150, 150), width=1)

    items = [
        ("Truffle Fries Starter", "1", "$12.00"),
        ("Grilled Atlantic Salmon", "2", "$48.00"),
        ("Wild Mushroom Risotto", "1", "$24.00"),
        ("Sparkling Mineral Water", "2", "$8.00"),
        ("Artisan Tiramisu Dessert", "1", "$11.00"),
    ]

    y = 280
    for desc, qty, amt in items:
        draw.text((40, y), desc, fill=(30, 30, 30), font=font_sm)
        draw.text((310, y), qty, fill=(30, 30, 30), font=font_sm)
        draw.text((w - 95, y), amt, fill=(30, 30, 30), font=font_sm)
        y += 28

    draw.line([(35, y + 10), (w - 35, y + 10)], fill=(150, 150, 150), width=1)
    y += 25

    draw.text((w - 240, y), "SUBTOTAL:", fill=(50, 50, 50), font=font_sm)
    draw.text((w - 95, y), "$103.00", fill=(30, 30, 30), font=font_sm)
    y += 22

    draw.text((w - 240, y), "SERVICE CHG (10%):", fill=(50, 50, 50), font=font_sm)
    draw.text((w - 95, y), "$10.30", fill=(30, 30, 30), font=font_sm)
    y += 22

    draw.text((w - 240, y), "STATE TAX (8%):", fill=(50, 50, 50), font=font_sm)
    draw.text((w - 95, y), "$8.24", fill=(30, 30, 30), font=font_sm)
    y += 28

    draw.line([(w - 250, y), (w - 35, y)], fill=(20, 20, 20), width=2)
    y += 8
    draw.text((w - 240, y), "TOTAL DUE:", fill=(20, 20, 20), font=font_lg)
    draw.text((w - 120, y), "$121.54", fill=(210, 30, 30), font=font_lg)
    y += 30
    draw.line([(w - 250, y), (w - 35, y)], fill=(20, 20, 20), width=2)

    y += 40
    draw.text((w // 2 - 80, y), "*** GRATUITY GUIDE ***", fill=(80, 80, 80), font=font_sm)
    draw.text((w // 2 - 110, y + 20), "18% = $18.54 | 20% = $20.60 | 22% = $22.66", fill=(90, 90, 90), font=font_sm)
    draw.text((w // 2 - 60, y + 55), "Thank you for dining!", fill=(70, 70, 70), font=font_sm)

    img.save(target_path)
    return target_path

def draw_receipt_grocery(target_path: str):
    w, h = 480, 820
    img = Image.new("RGB", (w, h), color=(252, 252, 250))
    draw = ImageDraw.Draw(img)
    font_lg = get_font(18)
    font_md = get_font(13)
    font_sm = get_font(11)

    draw.text((w // 2 - 115, 30), "FRESHMART SUPERSTORE", fill=(20, 20, 20), font=font_lg)
    draw.text((w // 2 - 85, 58), "Store #104 - Austin, TX", fill=(70, 70, 70), font=font_sm)
    draw.text((w // 2 - 75, 78), "1-800-555-FRESH", fill=(90, 90, 90), font=font_sm)

    draw.line([(30, 100), (w - 30, 100)], fill=(160, 160, 160), width=1)

    draw.text((35, 115), "RECEIPT #: REC-77401", fill=(20, 20, 20), font=font_md)
    draw.text((35, 138), "DATE:      2026-10-28 04:45 PM", fill=(50, 50, 50), font=font_sm)
    draw.text((35, 158), "LANE:      04 - CASHIER: Rachel", fill=(50, 50, 50), font=font_sm)

    draw.line([(30, 185), (w - 30, 185)], fill=(160, 160, 160), width=1)
    draw.text((35, 195), "DESCRIPTION                      PRICE", fill=(60, 60, 60), font=font_sm)
    draw.line([(30, 215), (w - 30, 215)], fill=(160, 160, 160), width=1)

    items = [
        ("Organic Whole Milk 1 Gal", "$4.29"),
        ("Farm Fresh Brown Eggs 12pk", "$3.99"),
        ("Seedless Red Grapes 2.1 lb", "$5.23"),
        ("Multigrain Artisan Bread", "$2.89"),
        ("Greek Yogurt Plain 32oz", "$4.50"),
        ("Dark Roast Ground Coffee 12oz", "$8.99"),
    ]

    y = 230
    for desc, prc in items:
        draw.text((35, y), desc, fill=(30, 30, 30), font=font_sm)
        draw.text((w - 95, y), prc, fill=(30, 30, 30), font=font_sm)
        y += 28

    draw.line([(30, y + 10), (w - 30, y + 10)], fill=(160, 160, 160), width=1)
    y += 25

    draw.text((w - 230, y), "SUBTOTAL:", fill=(50, 50, 50), font=font_sm)
    draw.text((w - 95, y), "$29.89", fill=(30, 30, 30), font=font_sm)
    y += 22

    draw.text((w - 230, y), "MEMBER SAVINGS:", fill=(16, 185, 129), font=font_sm)
    draw.text((w - 95, y), "-$3.00", fill=(16, 185, 129), font=font_sm)
    y += 22

    draw.text((w - 230, y), "SALES TAX:", fill=(50, 50, 50), font=font_sm)
    draw.text((w - 95, y), "$1.88", fill=(30, 30, 30), font=font_sm)
    y += 28

    draw.line([(w - 240, y), (w - 30, y)], fill=(20, 20, 20), width=2)
    y += 8
    draw.text((w - 230, y), "TOTAL:", fill=(20, 20, 20), font=font_lg)
    draw.text((w - 110, y), "$28.77", fill=(210, 30, 30), font=font_lg)
    y += 28
    draw.line([(w - 240, y), (w - 30, y)], fill=(20, 20, 20), width=2)

    y += 35
    draw.text((w // 2 - 85, y), "ITEMS PURCHASED: 6", fill=(60, 60, 60), font=font_sm)
    draw.text((w // 2 - 110, y + 20), "LOYALTY CARD: *******4911", fill=(60, 60, 60), font=font_sm)
    draw.text((w // 2 - 95, y + 55), "Save more with FreshMart App!", fill=(80, 80, 80), font=font_sm)

    img.save(target_path)
    return target_path

def draw_receipt_electronics(target_path: str):
    w, h = 500, 840
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font_lg = get_font(18)
    font_md = get_font(13)
    font_sm = get_font(11)

    draw.text((w // 2 - 110, 30), "TECHZONE ELECTRONICS", fill=(20, 20, 20), font=font_lg)
    draw.text((w // 2 - 90, 58), "TechZone Flagship Store #42", fill=(60, 60, 60), font=font_sm)
    draw.text((w // 2 - 100, 78), "San Francisco, CA • (415) 555-8900", fill=(80, 80, 80), font=font_sm)

    draw.line([(35, 105), (w - 35, 105)], fill=(170, 170, 170), width=1)

    draw.text((40, 120), "INVOICE / RECEIPT: TZ-2026-339", fill=(20, 20, 20), font=font_md)
    draw.text((40, 145), "TRANSACTION DATE: 2026-10-28 02:15 PM", fill=(50, 50, 50), font=font_sm)
    draw.text((40, 165), "REGISTER: POS-02 | ASSOCIATE: Kevin", fill=(50, 50, 50), font=font_sm)

    draw.line([(35, 190), (w - 35, 190)], fill=(170, 170, 170), width=2)
    draw.text((40, 200), "SKU / ITEM DESCRIPTION            PRICE", fill=(60, 60, 60), font=font_sm)
    draw.line([(35, 220), (w - 35, 220)], fill=(170, 170, 170), width=1)

    items = [
        ("KB-889  Wireless Mechanical Keyboard", "$89.99"),
        ("MS-201  Ergonomic Optical Mouse", "$39.99"),
        ("HB-700  USB-C Multiport Hub 7-in-1", "$45.00"),
        ("WAR-02  2-Year Extended Device Warranty", "$19.99"),
    ]

    y = 235
    for desc, prc in items:
        draw.text((40, y), desc, fill=(30, 30, 30), font=font_sm)
        draw.text((w - 95, y), prc, fill=(30, 30, 30), font=font_sm)
        y += 32

    draw.line([(35, y + 10), (w - 35, y + 10)], fill=(170, 170, 170), width=1)
    y += 25

    draw.text((w - 230, y), "SUBTOTAL:", fill=(50, 50, 50), font=font_sm)
    draw.text((w - 95, y), "$194.97", fill=(30, 30, 30), font=font_sm)
    y += 22

    draw.text((w - 230, y), "STATE TAX (8.0%):", fill=(50, 50, 50), font=font_sm)
    draw.text((w - 95, y), "$15.60", fill=(30, 30, 30), font=font_sm)
    y += 28

    draw.line([(w - 240, y), (w - 35, y)], fill=(20, 20, 20), width=2)
    y += 8
    draw.text((w - 230, y), "TOTAL PAID:", fill=(20, 20, 20), font=font_lg)
    draw.text((w - 110, y), "$210.57", fill=(210, 30, 30), font=font_lg)
    y += 28
    draw.line([(w - 240, y), (w - 35, y)], fill=(20, 20, 20), width=2)

    y += 35
    draw.text((w // 2 - 95, y), "METHOD: MASTERCARD *8820", fill=(60, 60, 60), font=font_sm)
    draw.text((w // 2 - 80, y + 20), "RETURN POLICY: 30 DAYS", fill=(80, 80, 80), font=font_sm)
    draw.text((w // 2 - 70, y + 55), "techzone-electronics.com", fill=(80, 80, 80), font=font_sm)

    img.save(target_path)
    return target_path

def generate_all(samples_dir: str, web_samples_dir: str):
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(web_samples_dir, exist_ok=True)

    receipts = [
        ("receipt_cafe.png", draw_receipt_cafe),
        ("receipt_restaurant.png", draw_receipt_restaurant),
        ("receipt_grocery.png", draw_receipt_grocery),
        ("receipt_electronics.png", draw_receipt_electronics),
    ]

    for fname, draw_fn in receipts:
        p1 = os.path.join(samples_dir, fname)
        p2 = os.path.join(web_samples_dir, fname)
        draw_fn(p1)
        draw_fn(p2)
        print(f"Generated {fname} in samples/ and web/samples/")

if __name__ == "__main__":
    generate_all("samples", "web/samples")
