# Maintenance Terminology — KB-Aligned

This terminology sheet is restricted to concepts relevant to the equipment and maintenance information represented in the team's D01–D09 knowledge base.

| Term | Project meaning | KB relevance |
|---|---|---|
| Preventive maintenance | Planned maintenance intended to reduce deterioration/failure risk. | D09 |
| Predictive maintenance | Maintenance informed by condition/diagnostic information. | D09 |
| Reactive / breakdown maintenance | Maintenance associated with an equipment failure/breakdown. | D09, D06 |
| Inspection | Systematic checking of equipment condition/status. | D04, D09 |
| Condition monitoring | Monitoring equipment condition using appropriate observations/tests. | D06, D08, D09 |
| Maintenance frequency | Source-defined timing for a specific maintenance item. | D09; never treat as a universal interval |
| Fault / failure | An abnormal condition or equipment failure requiring investigation/action. | D03, D06, D09 |
| Transformer | Power/distribution transformer equipment used in the substation. | D04, D07, D08, D09 |
| Circuit breaker | Switching/protection equipment whose maintenance depends on breaker type and procedure. | D01, D05, D06, D09 |
| VCB / SF6 breaker | Circuit-breaker types explicitly distinguished in D09 maintenance material. | D09 |
| OCB | Oil circuit breaker; separately covered in D09 maintenance material. | D09 |
| CT | Current transformer / instrument transformer. | D01, D06, D09 |
| PT / VT | Voltage/potential transformer terminology used in the KB. | D01, D06, D09 |
| CVT | Capacitive voltage transformer; do not assume its maintenance is identical to every PT/VT. | D06, D09 |
| Surge arrester | Equipment used to limit surge overvoltage exposure. | D01, D06, D09 |
| Isolator / disconnector | Switching/isolation equipment. | D01, D05, D09 |
| Battery bank | DC source supporting substation control/protection/communication functions. | D02, D05, D09 |
| Control & Relay Panel | Control/protection panel covered directly by D09. | D01, D09 |
| Earthing system | Substation earthing arrangement relevant to safety and fault-current paths. | D02, D05, D09 |
| Switchgear | Switching/protection assemblies within the substation/switchyard context. | D01, D05, D06, D09 |
| Maintenance record | Traceable record of work, observations/tests and supporting evidence. | D03, D04, D05, D06, D09 |
| Evidence | Source-supported information used to answer or assess a maintenance query. | D01-D09 |

## Critical project distinction

**Observation ≠ diagnosis ≠ maintenance decision**

Example:
- Observation: “Visible oil staining is present.”
- Evidence-supported statement: “Oil leakage/condition is a maintenance consideration where the applicable source identifies it.”
- Diagnosis: “An internal transformer fault exists.” → **Not justified from an ordinary photograph alone.**
- Maintenance decision: “Repair immediately.” → **Requires the applicable procedure and authorized human decision.**

The team's vision/RAG system should therefore report what is supported by the image and KB, not manufacture an engineering diagnosis.

## Source IDs

Use the team's catalog IDs **D01–D09** as the source identifiers in structured data and RAG metadata.
