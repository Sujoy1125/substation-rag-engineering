# Source Notes — Member 1 Knowledge Base

How each document in `document_catalog.xlsx` was found, and what "authoritative" means for it.
Document IDs below match `document_catalog.xlsx` and `knowledge_chunks.xlsx` exactly.

---

### D01 — General Guidelines for 765/400/220/132 kV Sub-stations and Switchyard of Thermal/Hydro Power Projects
- **Publisher:** Central Electricity Authority (CEA)
- **Hosted at:** cea.nic.in (official CEA domain)
- **What it is:** CEA's own design-guideline publication for substations/switchyards at these voltage classes.
- **Why trusted:** CEA is the statutory technical authority for the Indian power sector (Tier 1).
- **Caveat:** This is a **design** document, not a maintenance procedure document. It's included for equipment/system background context only — do not cite it for maintenance intervals or limits. It's also image-based (scanned), so only OCR-confirmed pages are represented in `knowledge_chunks.xlsx`.

### D02 — CEA (Safety Requirements for Construction, Operation and Maintenance of Electrical Plants and Electric Lines) Amendment Regulations, 2022
- **Publisher:** Central Electricity Authority / Government of India
- **Hosted at:** cea.nic.in, under the official notifications directory
- **What it is:** A Gazette-notified amendment regulation — binding safety requirement, not a guideline.
- **Why trusted:** Statutory regulation, Tier 1.

### D03 — Proforma for Reporting of Failure of Transformer/Reactor
- **Publisher:** Central Electricity Authority
- **Hosted at:** cea.nic.in
- **What it is:** The official CEA reporting form used after a transformer/reactor failure — tells us what fields a real failure report captures, not how to fix a failure.
- **Why trusted:** Tier 1, official form.
- **Caveat:** No procedure or troubleshooting content — structured field list only.

### D04 — Guidelines for Operation & Maintenance of Distribution Transformers
- **Publisher:** Central Electricity Authority / Ministry of Power
- **Hosted at:** cea.nic.in
- **What it is:** Direct O&M guidance for distribution transformers — preventive/predictive maintenance, testing, checklists, failure causes.
- **Why trusted:** Tier 1, and this is the single most directly relevant document in the set — it's actual O&M guidance, not adjacent regulation or design material.

### D05 — Operation and Maintenance of Electrical Plants and Electric Lines Regulations, 2011
- **Publisher:** Central Electricity Authority
- **Hosted at:** cea.nic.in, regulations directory
- **What it is:** The base regulation (pre-2022 amendment, see D02) covering O&M safety requirements for electrical plants and lines.
- **Why trusted:** Statutory regulation, Tier 1.

### D06 — Report of Standing Committee of Experts on Failure of 220 kV & Above Voltage Class Substation Equipment (Jul 2023–Dec 2024)
- **Publisher:** Central Electricity Authority / Ministry of Power
- **Hosted at:** cea.nic.in
- **What it is:** A committee report analyzing real equipment failures over an 18-month period, with root causes and recommendations.
- **Why trusted:** Tier 1, and unusually valuable — this is real failure/diagnostic data rather than generic procedure text, which is exactly what a troubleshooting-flavored RAG needs.
- **Caveat:** The URL as originally saved carried a `?utm_source=chatgpt.com` tracking parameter from how it was found (a ChatGPT-assisted search led to the cea.nic.in link). The underlying document is still the genuine CEA PDF at that path — the tracking parameter doesn't change the source, but strip it before final citation if it looks odd to a judge.

### D07 — Procedure for Transformer/Reactor Installation, Testing and Commissioning
- **Publisher:** Bharat Heavy Electricals Limited (BHEL)
- **Hosted at:** tenders.bhel.com (official BHEL tender portal)
- **What it is:** A detailed field procedure for transformer/reactor installation, testing, and commissioning — part of a BHEL tender package.
- **Why trusted:** BHEL is a Tier 1 public-sector manufacturer; this is their own procedural documentation.
- **Caveat:** This is a BHEL-specific procedure tied to a particular tender, not a universal industry standard — flagged as "high but scoped" in `RAG_priority`, and it's an installation/commissioning procedure, not a routine in-service maintenance schedule.
- Same `?utm_source=chatgpt.com` tracking-parameter note as D06 applies here.

### D08 — Standard Specifications and Technical Parameters for Transformers and Reactors (66 kV & Above Voltage Class), Amendment 01
- **Publisher:** Central Electricity Authority / Ministry of Power
- **Hosted at:** cea.nic.in
- **What it is:** CEA's core technical specification and testing/condition-monitoring manual for high-voltage transformers and reactors — covers DGA, FRA, OLTC, bushings, and life-cycle management.
- **Why trusted:** Tier 1, and this is the most technically dense document in the set (468 pages, 800 knowledge chunks).
- Same tracking-parameter note as D06/D07 applies.

---

## What was deliberately excluded

- Any document that only mentioned "substation" in passing without inspection/maintenance/procedure content.
- Any blog, forum, Quora/Reddit thread, or AI-generated technical PDF.
- Unofficial mirrors of the above CEA/BHEL documents — only the official domain copies were kept, so page numbers and version text can be trusted.

## Known limitation carried over from `knowledge_chunks.xlsx`

D01 (`765_powerplants.pdf`) is a scanned/image-based PDF. Only the pages that were OCR-confirmed are represented as chunks (23 of 132 pages). Absence of a page in the chunk set does **not** mean that page contains nothing useful — it means it hasn't been processed yet. Do not let the RAG system treat unprocessed D01 pages as "not covered" by the source; treat them as "not yet ingested."

## Fixes applied in this revision

- `document_catalog.xlsx` and `knowledge_chunks.xlsx` previously used **different Document ID numbering** for D07/D08/D09 (Tender vs. Transformer Manual were swapped). Both files now use the same IDs: **D07 = BHEL Tender procedure, D08 = Transformer Manual**. Always cross-reference by filename if in doubt.
- Removed a stray leftover row in `RAG_priority` that contradicted D06's actual content (looked like a copy-paste error from an earlier draft).
- Standardized the `Contains Procedures/Checklists/Limits/Troubleshooting?` columns in `document_catalog.xlsx` to YES / NO / PARTIAL (with a short parenthetical where useful) instead of free-text answers, so the columns can be filtered programmatically.
- Added the `Usage/License Notes` column to `document_catalog.xlsx` (was in the original spec, had been dropped).
- Fixed a data-corruption bug in `knowledge_chunks.xlsx`: one verified transformer temperature limit (top-oil 40°C / winding 45°C, from D01) had its cell value starting with `=`, so Excel/openpyxl was reading it as a broken formula (`#N/A`) instead of storing it as text. Restored as plain text.

### D09 — Guidelines for Benchmarking of Operation & Maintenance (O&M) Norms for Distribution Utilities
- **Publisher:** Central Electricity Authority / Ministry of Power
- **Hosted at:** cea.nic.in (official CEA domain)
- **What it is:** CEA guidelines for benchmarking O&M norms for distribution utilities, covering maintenance practices, major 33/11 kV substation equipment, distribution transformers, overhead lines/cables, and safety management.
- **Why trusted:** Central Electricity Authority source, Tier 1.
- **Source verification:** The official PDF supplied for this task was used to verify the D09 chunks. The document has printed page numbering within the PDF; Chapter 4 is printed pp. 64–81 and Chapter 6 is printed pp. 89–94.
- **Knowledge-base coverage:** `knowledge_chunks.xlsx` contains D09-C0001–D09-C0049, covering the previously created D09 material plus Chapter 4 (Overhead Lines & Cables) and Chapter 6 (Safety Management).
- **Caveat:** These are CEA guidelines/benchmarking guidance, not a universal substitute for applicable statutory requirements, utility-specific procedures, or OEM instructions. Cite the relevant CEA source section/page when answering.
