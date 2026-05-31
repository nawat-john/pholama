# แผนงานโปรเจกต์: ย่อ LLM 3B ให้เป็นโมเดลเฉพาะทางกิจการนักศึกษา

> เป้าหมาย: เอา Typhoon 2 3B มาตัด vocabulary ภาษาอื่น + ตัด layer/parameter ที่ไม่จำเป็น
> แล้วปรับให้เชี่ยวเฉพาะงานถาม-ตอบกิจการนักศึกษา (ไทย-อังกฤษ) จนรันได้บนเครื่องเล็ก/CPU
>
> ผลลัพธ์เป้าหมาย: **3.21B → ~2.1B พารามิเตอร์ → รันได้ใน ~1.5-2 GB RAM (หลัง quantize Q4)**

---

## 0. สเปกโมเดลฐาน (ตัวเลขที่ใช้คำนวณทั้งหมด)

ตรวจสอบยืนยันจาก `config.json` ของ Typhoon 2 3B ก่อนเริ่ม ค่ามาตรฐานของ Llama 3.2 3B:

| พารามิเตอร์ | ค่า |
|---|---|
| hidden_size (d) | 3,072 |
| num_hidden_layers | 28 |
| num_attention_heads | 24 |
| num_key_value_heads (GQA) | 8 |
| head_dim | 128 |
| intermediate_size (FFN) | 8,192 |
| vocab_size | 128,256 |
| tie_word_embeddings | true |
| **รวมพารามิเตอร์** | **~3.21B** |

### พารามิเตอร์อยู่ที่ไหนบ้าง (สำคัญมากต่อการวางแผนตัด)

| ส่วนประกอบ | พารามิเตอร์ | สัดส่วน |
|---|---|---|
| Embedding (tied input+output) | 128,256 × 3,072 = **394M** | 12.3% |
| 28 transformer layers | **2,819M** | 87.7% |
| — ต่อ 1 layer | ~100.7M | |
| —— Attention ต่อ layer | ~25.2M (q+k+v+o proj) | 25% ของ layer |
| —— MLP/FFN ต่อ layer | ~75.5M (gate+up+down) | 75% ของ layer |

**ข้อสังเกตสำคัญ 2 ข้อ:**
1. ใน 3B นี้ embedding เป็นแค่ ~12% (ต่างจากโมเดลจิ๋ว 0.5-1B ที่ embedding อาจเกิน 40%) ดังนั้นการตัด vocab ช่วยได้จริงแต่ % ไม่เยอะเท่าโมเดลเล็ก — พลังการลดขนาดที่แท้จริงอยู่ที่ structured pruning ของ transformer layers
2. MLP/FFN กินถึง 75% ของแต่ละ layer → ถ้าจะตัด width ให้เน้นตัด FFN

---

## Phase 1 — เตรียมสภาพแวดล้อมและ baseline (≈1 สัปดาห์)

**เป้าหมาย:** ตั้งเครื่องมือให้พร้อมและวัด baseline ของโมเดลเดิมไว้เปรียบเทียบ

**ขั้นตอนย่อย:**
- ติดตั้ง: PyTorch, Hugging Face `transformers` + `peft` + `bitsandbytes` + `datasets`, `lm-eval-harness`, `llama.cpp`
- โหลด Typhoon 2 3B แล้วยืนยัน config จริง นับพารามิเตอร์ด้วย `sum(p.numel() for p in model.parameters())`
- รัน baseline eval: วัดคุณภาพในโดเมน (test set กิจการนักศึกษา), วัดความสามารถนอกโดเมน (โค้ด/คณิต — ไว้ดูว่าตกลงแค่ไหนหลัง specialize), วัด latency/RAM/ขนาดไฟล์
- ตั้ง Git repo + บันทึก experiment (เช่น ใช้ Weights & Biases ฟรี หรือไฟล์ CSV ง่ายๆ)

**Deliverable:** ตาราง baseline (คุณภาพในโดเมน, นอกโดเมน, RAM, latency, ขนาด) + repo ที่ reproduce ได้

**พารามิเตอร์/RAM:** 3.21B; BF16 ≈ 6.4 GB weights (~7-8 GB RAM ตอน inference)

**Compute:** Colab/Kaggle free GPU (T4/P100 16GB) พอ

---

## Phase 2 — สร้างชุดข้อมูลโดเมน (≈3-4 สัปดาห์ — ขั้นที่หนักและสำคัญที่สุด)

**เป้าหมาย:** ได้ชุด Q&A กิจการนักศึกษา ไทย-อังกฤษ คุณภาพสูง ~2,000-10,000 คู่ ที่จะถูกใช้ซ้ำ 3 ที่ (calibration, recovery training, evaluation)

**ขั้นตอนย่อย:**
- รวบรวมแหล่งจริง: ระเบียบ/ประกาศ/FAQ/คู่มือนักศึกษาของมหาวิทยาลัย, หน้าเว็บกิจการนักศึกษา, เอกสารทุน/หอพัก/กิจกรรม
- แปลงเป็นคู่ถาม-ตอบด้วย LLM ตัวใหญ่ (synthetic generation) โดยออกแบบ prompt ให้สร้างคำถามหลากหลายสไตล์ (ทางการ/ไม่ทางการ, ไทย/อังกฤษ/ปนกัน)
- คัดกรองด้วยมือ: ตัดคำตอบผิด/กำกวม/ซ้ำ ออก — คุณภาพสำคัญกว่าปริมาณ
- แบ่ง train/validation/test ตั้งแต่แรก (เช่น 80/10/10) และ **ห้ามให้ test รั่วไปอยู่ใน calibration/training**
- เก็บ corpus ดิบ (ข้อความไทย-อังกฤษล้วน) แยกไว้ด้วย สำหรับใช้นับ token ใน Phase 3

**Deliverable:** dataset เวอร์ชัน v1 (train/val/test) + corpus สำหรับ vocab analysis + เอกสารอธิบายวิธีสร้าง

**ความเสี่ยง:** ดาตาน้อย/คุณภาพต่ำ = ทั้งโปรเจกต์พัง เผื่อเวลาขั้นนี้ให้มากที่สุด

**พารามิเตอร์/RAM:** ยังไม่เปลี่ยน (3.21B)

---

## Phase 3 — ตัด Vocabulary (Vocabulary Trimming) (≈1 สัปดาห์)

**เป้าหมาย:** ลบ token ภาษาอื่นที่ไม่ใช้ (dead tokens) ออกจากเมทริกซ์ embedding — ส่วนที่ "ตัดภาษาอื่น" ได้ตรงตัวที่สุด

**หลักการ:** อ้างอิงเปเปอร์ *Efficient Multilingual Language Model Compression through Vocabulary Trimming* (Ushio et al., EMNLP 2023) — โดยทั่วไปเหลือ vocab ~50% ก็รักษาคุณภาพเดิมได้ และยังช่วยลด social bias

**ขั้นตอนย่อย:**
- รัน tokenizer บน corpus ไทย-อังกฤษทั้งหมด นับความถี่ของทุก token
- เลือก token ที่จะเก็บ: token ที่ปรากฏในข้อมูล + token พิเศษ (BOS/EOS/PAD ฯลฯ) + ตัวเลข/เครื่องหมายพื้นฐาน เผื่อ threshold ความถี่ขั้นต่ำ
- ลบแถวของ dead tokens ออกจากทั้ง `embed_tokens` และ `lm_head` (ในที่นี้ tied กัน เลยจัดการชุดเดียว)
- สร้าง mapping `old_id → new_id` และบันทึก tokenizer ใหม่
- ทดสอบ sanity: encode/decode ข้อความไทย-อังกฤษต้องได้ผลเดิม

**Deliverable:** โมเดล vocab-trimmed + tokenizer ใหม่ + รายงานว่าตัด token ไปกี่ตัว

**การคำนวณ (สมมติเก็บ vocab 128,256 → ~40,000 token):**
- ตัดออก ~88,000 token × 3,072 = **~271M พารามิเตอร์**
- embedding: 394M → ~123M
- **รวมโมเดล: 3.21B → ~2.94B** (ลด ~8%)

**Compute:** CPU ล้วน ไม่ต้องใช้ GPU

---

## Phase 4 — Structured Pruning (≈2-3 สัปดาห์ — แกนของโปรเจกต์)

**เป้าหมาย:** ตัด layer/head/channel ที่สำคัญน้อยออก เพื่อลดขนาด transformer (87.7% ของโมเดล) โดยใช้ข้อมูลโดเมนเป็น calibration

**กลยุทธ์ (เริ่มจากง่ายไปยาก):**

### 4a. Layer pruning (แนะนำเริ่มที่นี่)
ตัด transformer block ทั้งชั้นที่สำคัญน้อย — อ้างอิง ShortGPT / Shortened LLaMA (ใช้ Taylor expansion หรือ perplexity เป็นเกณฑ์) งานวิจัยพบว่าตัดได้ราว ~30% ของ layer โดยยังคุมคุณภาพ
- วัดความสำคัญของแต่ละ layer ด้วย calibration data (ข้อมูลโดเมน)
- ตัด 8 layer ที่สำคัญน้อยสุด: 28 → 20 layers

### 4b. Width pruning (ทางเลือกเพื่อย่อเพิ่ม)
ตัด attention head และ/หรือลด FFN intermediate dim — ใช้เครื่องมือ **LLM-Pruner** (gradient-based importance) หรือ repo **LLM-Pruning Collection** ที่รวม ShortGPT/Wanda/SparseGPT/Sheared LLaMA/LLM-Pruner สำหรับตระกูล Llama
- เช่น ลด FFN 8,192 → 6,144 (−25%) บน layer ที่เหลือ

**จุดสำคัญ:** ใช้ **ชุดข้อมูลกิจการนักศึกษาเป็น calibration data** — กลไกนี้แหละที่ทำให้โมเดลเอนเอียงไปทางโดเมนและตัดส่วนที่รับใช้โค้ด/คณิตทิ้งโดยปริยาย

**การคำนวณ — 2 สถานการณ์ (ต่อจาก Phase 3 = embedding 123M):**

| สถานการณ์ | วิธี | transformer | รวม |
|---|---|---|---|
| Moderate | layer 28→20 เท่านั้น | 20 × 100.7M = 2,013M | **~2.14B** |
| Aggressive | layer 28→20 + FFN −25% | ~1,636M | **~1.76B** |

(Moderate: ตัด 8 layer = −806M; Aggressive: ตัด FFN เพิ่มอีก ~377M)

**ความเสี่ยง:** ยิ่งตัดเยอะ คุณภาพยิ่งตก และต้องใช้ข้อมูล recovery มากขึ้น เริ่มที่ Moderate ก่อน ค่อยลองดันไป Aggressive แล้วเทียบ

**Compute:** Colab/Kaggle free GPU พอสำหรับ 3B

---

## Phase 5 — Recovery Fine-tune (≈1-2 สัปดาห์ — ห้ามข้าม)

**เป้าหมาย:** กู้คุณภาพในโดเมนที่ตกหลัง prune กลับมา (ความสามารถนอกโดเมนจะไม่ฟื้น = สิ่งที่ต้องการ)

**ขั้นตอนย่อย:**
- ใช้ **QLoRA** (LoRA บน base ที่ quantize 4-bit) fine-tune บน train set โดเมน
- ตั้ง seq len ≤ 2,048, batch เล็ก + gradient accumulation เพื่อให้พอดี free GPU
- เลือก hyperparameter LoRA (rank, alpha, target modules), monitor val loss
- เมื่อพอใจแล้ว merge LoRA adapter กลับเข้า weight (พารามิเตอร์เพิ่มแทบไม่นับ)
- รัน eval ซ้ำเทียบกับ baseline

**Deliverable:** โมเดล pruned + recovered + ตารางเทียบคุณภาพก่อน/หลัง recovery

**พารามิเตอร์:** ไม่เปลี่ยน (~2.1B สำหรับ Moderate) — แต่คุณภาพในโดเมนกลับมา

**RAM ตอนเทรน (QLoRA บน ~2-3B):** base 4-bit ~1.2-1.9 GB + adapter/optimizer/activations → peak ~6-12 GB ขึ้นกับ batch×seq → **พอดีกับ free T4/P100 16GB**

---

## Phase 6 — Quantization (≈2-3 วัน)

**เป้าหมาย:** บีบขนาด/RAM สุดท้ายด้วยการลดความละเอียดตัวเลข

**ขั้นตอนย่อย:**
- แปลงเป็น **GGUF** ผ่าน llama.cpp (สำหรับรัน CPU/เครื่องเล็ก) — เลือกระดับ เช่น Q4_K_M (สมดุลดี), Q5_K_M (คุณภาพสูงขึ้น), Q8 (เกือบเท่า fp16)
- หรือ GPTQ/AWQ ถ้าจะรันบน GPU เล็ก
- เทียบคุณภาพแต่ละระดับ quantization กับ Q8/fp16 เพื่อหาจุดสมดุล

**พารามิเตอร์:** เท่าเดิม แต่ขนาด/RAM ลดตามความละเอียด (ดูตารางสรุปด้านล่าง)

**Compute:** CPU ได้

---

## Phase 7 — Deploy + ประเมินผล + เขียนรายงาน (≈2 สัปดาห์)

**สถาปัตยกรรมแนะนำ — อย่าฝังความรู้ไว้ใน weight อย่างเดียว:**
- ใช้ **RAG**: โมเดลที่ผ่าน pipeline = เครื่องมือเข้าใจภาษา/ให้เหตุผล; ข้อเท็จจริง (กำหนดการ/ระเบียบที่เปลี่ยนทุกปี) ดึงจากฐานเอกสารผ่าน retrieval
- serve ด้วย **Ollama** (ง่ายสุด) + vector DB เช่น FAISS/Chroma + embedding model เล็ก

**การประเมินผล (3 แกน — ส่วนที่ทำให้พอร์ตดูเป็น expert):**
1. คุณภาพในโดเมน: เทียบ accuracy/quality บน test กิจการนักศึกษา (ก่อน vs หลัง)
2. การถดถอยนอกโดเมน: วัดว่าความสามารถโค้ด/คณิตตกลงจริง (พิสูจน์ว่า specialization สำเร็จ)
3. ประสิทธิภาพ: ขนาดไฟล์, RAM, latency, tokens/วินาที

ใช้ **lm-eval-harness** เป็นมาตรฐานการวัด

**Deliverable:** ระบบ demo ที่รันได้ + รายงานเปรียบเทียบครบ 3 แกน + README

---

## ตารางสรุป: ขนาดและ RAM ที่ระดับ quantization ต่างๆ

ขนาด weights = พารามิเตอร์ × bytes/param  •  RAM ตอน inference ≈ weights + KV cache + overhead (~0.3-0.7 GB)

| ระดับ | bytes/param | โมเดลเดิม 3.21B | **เป้าหมาย Moderate 2.1B** | Aggressive 1.76B |
|---|---|---|---|---|
| BF16 | 2.0 | 6.4 GB | 4.2 GB | 3.5 GB |
| INT8 / Q8 | 1.0 | 3.2 GB | 2.1 GB | 1.8 GB |
| Q5_K_M | ~0.69 | 2.2 GB | 1.45 GB | 1.2 GB |
| **Q4_K_M** | ~0.58 | 1.86 GB | **1.22 GB** | 1.0 GB |

**บรรทัดสรุป (เป้าหมาย Moderate + Q4_K_M):**
- weights ~1.22 GB + KV cache ~0.34 GB (ที่ 4K context) + overhead ~0.3 GB → **RAM รวม ~1.5-2 GB**
- รันได้บนแล็ปท็อปทั่วไป, mini PC, หรือแม้แต่อุปกรณ์ระดับสมาร์ตโฟน/Raspberry Pi class

### หมายเหตุ KV cache
ต่อ 1 token (fp16) = 2 × layers × KV_heads × head_dim × 2 bytes
- เดิม 28 layers: ~0.11 MB/token → 4K context ≈ 0.47 GB
- pruned 20 layers: ~0.08 MB/token → 4K context ≈ 0.34 GB
- quantize KV cache ได้อีกถ้าต้องการลด RAM เพิ่ม

---

## ไทม์ไลน์รวม (part-time ~10-15 ชม./สัปดาห์)

| สัปดาห์ | Phase |
|---|---|
| 1 | Phase 1 — setup + baseline |
| 2-5 | Phase 2 — สร้างชุดข้อมูล (หนักสุด) |
| 6 | Phase 3 — vocab trimming |
| 7-9 | Phase 4 — structured pruning (ทดลองหลายค่า) |
| 10-11 | Phase 5 — recovery fine-tune |
| 11 | Phase 6 — quantization |
| 12-13 | Phase 7 — deploy + eval + รายงาน |

รวม ~2.5-3 เดือน

---

## Baseline เปรียบเทียบ (สิ่งที่ expert ต้องมี)

ออกแบบให้มี 3 baseline เสมอ เพื่อพิสูจน์ว่า pipeline คุ้มค่า:
1. โมเดลเดิม (Typhoon 2 3B) — ไม่ทำอะไร
2. โมเดลเดิม + QLoRA fine-tune + quantize (ไม่ prune)
3. Pipeline เต็ม (vocab trim + prune + recovery + quantize)

คำถามที่ต้องตอบในรายงาน: **pipeline เต็มได้ขนาด/ความเร็วดีกว่า baseline ข้อ 2 จริงไหม และแลกมาด้วยคุณภาพเท่าไหร่?** การมี baseline นี่แหละที่แยกงานวิจัยจริงออกจากงานทำตามกระแส

---

## สรุปเครื่องมือ

| งาน | เครื่องมือ |
|---|---|
| โหลด/แก้โมเดล | Hugging Face Transformers |
| Vocab trimming | สคริปต์เอง (อ้างอิงเปเปอร์ arXiv 2305.15020) |
| Structured pruning | LLM-Pruner, LLM-Pruning Collection, ShortGPT |
| Recovery fine-tune | PEFT (QLoRA) + bitsandbytes |
| Quantization | llama.cpp (GGUF), GPTQ/AWQ |
| Evaluation | lm-eval-harness |
| Deploy | Ollama + FAISS/Chroma (RAG) |
| Compute | Colab / Kaggle free GPU + CPU |
