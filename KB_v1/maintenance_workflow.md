# Maintenance Workflow — KB-Aligned Member 2 Reference

## Scope

This workflow is a **research-derived domain model for the SIH system**, aligned to the documents actually present in the team's final knowledge base (D01–D09).

It is **not** a claim that every utility follows one identical sequence. Actual maintenance must follow the applicable CEA regulation/guideline, utility procedure, manufacturer instruction, permit/clearance process and approved test method.

## Evidence in the supplied KB

- **D02 / D05** provide the CEA safety and O&M regulatory baseline for electrical plants, lines and substations.
- **D04** provides direct operation, maintenance, testing and failure-related material for distribution transformers.
- **D06** provides reported failure evidence and recommendations for 220 kV+ substation equipment.
- **D07 / D08** provide transformer installation/testing/commissioning and transformer technical/specification context.
- **D09** provides equipment-specific O&M practices, frequencies and fault/remedy information for multiple distribution-substation equipment categories.
- **D01** provides substation/switchyard equipment and design context.

## Domain workflow

### 1. Identify the equipment and scope
Record the equipment type, asset/equipment ID where available, location, manufacturer/model where available, and the applicable document/version.

### 2. Establish the applicable safety and authorization conditions
The system should surface relevant safety requirements from the applicable CEA/utility material. It should **not authorize or control electrical work**.

### 3. Inspect
Record what is actually observed:
- visible condition
- indicator/status information
- physical damage/corrosion/leakage where visible
- relevant equipment-specific inspection points

Keep observation separate from diagnosis.

### 4. Measure or test where the source/procedure requires it
Record the measurement/test result together with:
- unit
- method/test type
- date
- applicable equipment
- source document/version/page

Do not invent a test limit when the KB does not provide one.

### 5. Compare against documented criteria
Retrieve the applicable criterion, schedule or procedure from the authoritative source and respect its scope.

For D09, **maintenance frequencies are item-specific**. They must not be converted into one universal interval for an equipment category.

### 6. Decide the maintenance response
A qualified/authorized person determines the action, such as:
- continue monitoring
- planned maintenance
- corrective maintenance
- additional testing/diagnostics
- escalation

The chatbot should support retrieval and evidence organization, not make the final operational decision.

### 7. Human verification
Record the responsible person, action, verification status and supporting evidence.

### 8. Close the maintenance record
Store the maintenance result, observations, test results, linked source/version/page and relevant photographs/reports so the event is auditable.

## RAG-specific mapping

Query
→ identify equipment/topic
→ retrieve KB evidence
→ check document scope/version
→ check evidence sufficiency
→ if sufficient: answer + source citation
→ if insufficient/ambiguous/out-of-scope: ask for missing context or refuse
→ if operational action is requested: escalate to authorized maintenance workflow
→ store evidence/reference for audit

## D09 schedule mapping used in the equipment inventory

- Power Transformer — D09-C0009, PDF pp. 22–26
- Circuit Breaker — D09-C0010 and D09-C0011, PDF pp. 30–35
- CT — D09-C0012, PDF pp. 36–37
- PT/VT — D09-C0013, PDF pp. 37–38
- Surge Arrester — D09-C0014, PDF p. 39
- Isolator/Disconnector — D09-C0015, PDF p. 40
- Battery Bank — D09-C0017, PDF pp. 42–43
- Control & Relay Panel — D09-C0018, PDF pp. 44–45
- Earthing System — D09-C0019, PDF pp. 46–47

For **Busbar, standalone Protection Relay, and generic Switchgear**, the inventory leaves the D09 frequency as **NOT VERIFIED** because the current KB does not contain a single universal D09 schedule for those categories.

## Sources — supplied KB only

D01 — CEA, General Guidelines for 765/400/220/132 kV Sub-stations and Switchyard of Thermal/Hydro Power Projects  
D02 — CEA, Safety Requirements for Construction, Operation and Maintenance of Electrical Plants and Electric Lines, Amendment Regulations 2022  
D03 — CEA, Proforma for Reporting of Failure of Transformer/Reactor  
D04 — CEA, Operation & Maintenance of Distribution Transformers  
D05 — CEA, Operation and Maintenance of Electrical Plants and Electric Lines Regulations, 2011  
D06 — CEA, Report of Standing Committee of Experts on Failure of 220 kV & Above Voltage Class Substation Equipment  
D07 — BHEL, Procedure for Transformer/Reactor Installation, Testing and Commissioning  
D08 — CEA, Standard Specifications and Technical Parameters for Transformers and Reactors (66 kV & Above)  
D09 — CEA, Guidelines for Benchmarking of Operation & Maintenance (O&M) Norms for Distribution Utilities, 2024
