# Dataset Card — reg.chula Q&A (v1)

**โดเมน:** สำนักงานการทะเบียน จุฬาลงกรณ์มหาวิทยาลัย (https://www.reg.chula.ac.th/th/)
**ภาษา:** ไทย / อังกฤษ / ปนกัน — ใช้คำว่า "นิสิต"

## ที่มา
สังเคราะห์ด้วย Gemini (multi-model fallback: gemini-2.5-flash-lite / flash-lite-latest /
flash-latest / 2.5-flash — สลับเมื่อโควต้าฟรีรายวันหมด) จากข้อความที่ crawl จากเว็บ
reg.chula.ac.th (หน้า HTML + PDF ระเบียบ/ประกาศ) คำตอบ grounded ในเนื้อหาเว็บเท่านั้น

## จำนวน
- raw ที่ gen ได้: 4198
- หลัง heuristic filter: 4144
- หลัง dedup (clean): 4016
- train / val / test: 3212 / 401 / 403 (80/10/10, seed=42)

## การกระจาย (clean)
- lang: {'th': 2569, 'en': 1278, 'mix': 169}
- style: {'casual': 1218, 'formal': 2789, 'mix': 9}
- จำนวน source page/pdf: 81

## การใช้งานในไปป์ไลน์
- calibration (Phase 4) + recovery training (Phase 5): ใช้ **train** เท่านั้น
- evaluation (Phase 1/7): ใช้ **test**
- ⚠️ test แยกเด็ดขาด ห้ามรั่วเข้า calibration/training (ตรวจแล้วผ่าน)

## ข้อจำกัด / ที่ควรทำต่อ
- เป็นการคัด "อัตโนมัติ" — แผนแนะนำคัดด้วยมืออีกชั้น (เปิด qa_raw.jsonl ลบที่ผิด/กำกวม)
- เนื้อหาวันเวลา (ปีการศึกษา 2568/2569) เป็น snapshot ณ วันที่ crawl — Phase 7 ใช้ RAG ดึงของจริงแทน
